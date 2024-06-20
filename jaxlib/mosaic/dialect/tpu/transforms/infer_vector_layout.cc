/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/layout.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_INFERVECTORLAYOUTPASS
#define GEN_PASS_DEF_INFERVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

using ImplicitDim = VectorLayout::ImplicitDim;

static constexpr int kLayoutLog = 10;

class Print {
 public:
  explicit Print(Operation *t) : payload_(t) {}
  Operation *payload_;

 private:
  friend std::ostream &operator<<(std::ostream &, Print);
};

std::ostream &operator<<(std::ostream &os, Print p) {
  std::string s;
  llvm::raw_string_ostream tmp_os(s);
  p.payload_->print(tmp_os);
  os << tmp_os.str();
  return os;
}

bool is_fully_replicated(const Layout &layout) {
  static LayoutOffsets replicated_offsets = {std::nullopt, std::nullopt};
  return layout.has_value() && layout->offsets() == replicated_offsets;
}

TiledLayoutAttr getMemRefLayout(Value ref) {
  if (auto erase_op = ref.getDefiningOp<tpu::EraseLayoutOp>()) {
    ref = erase_op.getOperand();
  }
  return cast<TiledLayoutAttr>(cast<MemRefType>(ref.getType()).getLayout());
}

LogicalResult verifyDivisibleIndex(Value tiled_index, int64_t tiling, int dim,
                                   Operation *op) {
  if (!isGuaranteedDivisible(tiled_index, tiling)) {
    return op->emitOpError("cannot statically prove that index in dimension ")
           << dim << " is a multiple of " << tiling;
  }
  return success();
}

// TODO(apaszke): Test that this pass fills in NoLayout for all operations that
// have corresponding native instructions.
class VectorLayoutInferer {
 public:
  explicit VectorLayoutInferer(std::array<int64_t, 2> target_shape)
      : target_shape_({target_shape[0], target_shape[1]}),
        default_tiling_(target_shape) {}

#define TPU_CHECK_OP(cond, msg) \
  if (!(cond)) {                \
    op->emitOpError(msg);       \
    return failure();           \
  }

#define NYI(msg)                            \
  op->emitOpError("not implemented: " msg); \
  return failure();

  LogicalResult inferBlock(
      Block &block,
      const std::function<LogicalResult(Operation *)> &match_terminator) {
    for (Operation &any_op : block.without_terminator()) {
      VLOG(kLayoutLog) << Print(&any_op);
      if (any_op.hasAttr("in_layout") || any_op.hasAttr("out_layout")) {
        if (auto op = dyn_cast<tpu::AssumeLayoutOp>(any_op)) {
          TPU_CHECK_OP(
              any_op.hasAttr("in_layout") && any_op.hasAttr("out_layout"),
              "expect layout attributes in tpu::AssumeLayoutOp");
          continue;
        } else {
          any_op.emitOpError("layout attributes already attached");
          return failure();
        }
      }
      bool has_vector_io = false;
      for (auto op : any_op.getOperands()) {
        has_vector_io |= op.getType().isa<VectorType>();
      }
      for (auto r : any_op.getResults()) {
        has_vector_io |= r.getType().isa<VectorType>();
      }
      if (!has_vector_io && any_op.getRegions().empty()) {
        SmallVector<Layout, 4> in_layout(any_op.getNumOperands(), kNoLayout);
        if (any_op.getNumResults() == 0) {
          setInLayout(&any_op, in_layout);
        } else if (any_op.getNumResults() == 1) {
          setLayout(&any_op, in_layout, kNoLayout);
        } else {
          any_op.emitOpError("Multi-result ops not supported");
          return failure();
        }
      } else if (isa<arith::ExtFOp, arith::ExtSIOp>(any_op)) {
        if (inferExt(&any_op).failed()) {
          return failure();
        }
      } else if (isa<arith::TruncFOp, arith::TruncIOp>(any_op)) {
        if (inferTrunc(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::SelectOp>(any_op)) {
        auto true_ty = dyn_cast<VectorType>(op.getTrueValue().getType());
        auto false_ty = dyn_cast<VectorType>(op.getFalseValue().getType());
        TPU_CHECK_OP(static_cast<bool>(true_ty) == static_cast<bool>(false_ty),
                     "Only one side of arith is a vector?");
        if (true_ty) {
          TPU_CHECK_OP(true_ty.getElementTypeBitWidth() == kNativeBitwidth &&
                           false_ty.getElementTypeBitWidth() == kNativeBitwidth,
                       "Only 32-bit select supported");
        }
        if (inferElementwise(&any_op, /*check_bitwidth=*/false).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ExtUIOp>(any_op)) {
        auto in_ty = dyn_cast<VectorType>(op.getIn().getType());
        auto out_ty = dyn_cast<VectorType>(op.getType());
        TPU_CHECK_OP(static_cast<bool>(in_ty) == static_cast<bool>(out_ty),
                     "Input and output are not both vectors?");
        if (in_ty) {
          TPU_CHECK_OP(in_ty.getElementTypeBitWidth() == 1,
                       "Only extending i1 is supported");
        }
        if (inferElementwise(&any_op, /*check_bitwidth=*/false).failed()) {
          return failure();
        }
      } else if (isa<arith::CmpIOp>(any_op) || isa<arith::CmpFOp>(any_op)) {
        Operation *op = &any_op;  // For TPU_CHECK_OP macros, which use the `op`
                                  // variable in scope
        auto lhs_ty = dyn_cast<VectorType>(any_op.getOperand(0).getType());
        auto rhs_ty = dyn_cast<VectorType>(any_op.getOperand(1).getType());
        TPU_CHECK_OP(static_cast<bool>(lhs_ty) == static_cast<bool>(rhs_ty),
                     "Only one side of cmp is a vector?");
        // TODO(tlongeri): Check that TPU generation supports comparison.
        if (inferElementwise(&any_op, /*check_bitwidth=*/false).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ConstantOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<cf::AssertOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<memref::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::IfOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::ForOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::WhileOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RotateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::DynamicRotateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::ConcatenateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StridedLoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StridedStoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::MatmulOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::EraseLayoutOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::IotaOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::GatherOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::BitcastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RepeatOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::TraceOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::PRNGRandomBitsOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RegionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::BroadcastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ContractionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ExtractOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::MultiDimReductionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ShapeCastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::StoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::TransposeOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ExtractStridedSliceOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (OpTrait::hasElementwiseMappableTraits(&any_op)) {
        // We put elementwise rule to the end in case the overriding rule.
        if (inferElementwise(&any_op).failed()) {
          return failure();
        }
      } else {
        any_op.emitOpError("unsupported in vector layout inference");
        return failure();
      }
      CHECK(any_op.getNumResults() == 0 || any_op.hasAttr("out_layout"));
      CHECK(any_op.getNumOperands() == 0 || any_op.hasAttr("in_layout"));
    }
    return match_terminator(block.getTerminator());
  }

  LogicalResult infer(arith::ConstantOp op) {
    if (op.getType().isSignlessIntOrIndexOrFloat()) {
      setOutLayout(op, kNoLayout);
      return success();
    }
    if (auto ty = dyn_cast<VectorType>(op.getType())) {
      auto elems = dyn_cast<DenseElementsAttr>(op.getValue());
      TPU_CHECK_OP(ty.getElementType().isSignlessIntOrIndexOrFloat(),
                   "expected scalar element type in vector");
      TPU_CHECK_OP(ty.getRank() > 0, "rank 0 vectors unsupported");
      TPU_CHECK_OP(elems, "expected vector constants to use DenseElementsAttr");
      auto bitwidth = ty.getElementTypeBitWidth();
      if (elems.isSplat()) {
        if (ty.getRank() == 1) {
          // Here, we choose to lay out along lanes arbitrarily. It would be
          // equally valid to go with sublanes. Still, this value is so easy
          // to relayout that it shouldn't really make a difference.
          setOutLayout(op, VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                                        nativeTiling(bitwidth),
                                        ImplicitDim::kSecondMinor));
        } else {  // ty.getRank() >= 2
          setOutLayout(
              op, VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                               nativeTiling(bitwidth), ImplicitDim::kNone));
        }
      } else {
        TPU_CHECK_OP(ty.getElementTypeBitWidth() == kNativeBitwidth,
                     "Only 32-bit non-splat constants supported");
        if (ty.getRank() == 1) {
          if (ty.getDimSize(0) <= target_shape_[0]) {
            // Use 2D layout with replication.
            NYI("small 1D constants");
          } else {  // NOLINT(readability-else-after-return)
            NYI("large 1D constants");
          }
        } else {  // ty.getRank() >= 2
          setOutLayout(op, VectorLayout(kNativeBitwidth, {0, 0},
                                        default_tiling_, ImplicitDim::kNone));
        }
      }
      return success();
    }
    op.emitOpError("unsupported constant type");
    return failure();
  }

  LogicalResult infer(cf::AssertOp op) {
    setInLayout(op, {kNoLayout});
    return success();
  }

  LogicalResult infer(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return inferBlock(
        op.getBody().front(), [this](Operation *op) -> LogicalResult {
          TPU_CHECK_OP(isa<func::ReturnOp>(op),
                       "Expected func.return terminator");
          for (Value o : op->getOperands()) {
            TPU_CHECK_OP(!isa<VectorType>(o.getType()),
                         "vector returns unsupported");
          }
          SmallVector<Layout, 4> in_layout(op->getNumOperands(), {kNoLayout});
          setInLayout(op, in_layout);
          return success();
        });
  }

  LogicalResult infer(memref::LoadOp op) {
    TPU_CHECK_OP(op.getType().isSignlessIntOrIndexOrFloat(),
                 "memref.load with non-scalar result");
    SmallVector<Layout, 5> in_layout(op.getNumOperands(), {kNoLayout});
    setLayout(op, in_layout, kNoLayout);
    return success();
  }

  LogicalResult infer(scf::IfOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 1, "expected one operand");
    setInLayout(op, {kNoLayout});
    if (inferBlock(*op.thenBlock(), match_yield).failed()) {
      op.emitOpError("failed to infer layout for then branch");
      return failure();
    }
    auto then_yield = op.thenBlock()->getTerminator();
    TPU_CHECK_OP(then_yield->getOperandTypes() == op->getResultTypes(),
                 "scf if results and then branch yield operands do not match");
    auto then_yield_in_layouts = getLayoutFromOperands(then_yield);
    if (auto else_block = op.elseBlock()) {
      if (inferBlock(*else_block, match_yield).failed()) {
        op.emitOpError("failed to infer layout for else branch");
        return failure();
      }
    }
    if (op->getNumResults() == 0) {
      return success();
    }
    // If the if op has results, it should have both then and else regions with
    // yield op.
    auto else_yield = op.elseBlock()->getTerminator();
    TPU_CHECK_OP(else_yield->getOperandTypes() == op->getResultTypes(),
                 "scf if results and else branch yield operands do not match");
    auto else_yield_in_layouts = getLayoutFromOperands(else_yield);
    // Find a compatible layout from then and else branches for each reuslt. For
    // example, if we yield offset (*, *) in then branch and offset (*, 0) in
    // else branch, the result offset should be (*, 0).
    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    int out_idx = 0;
    for (auto [then_layout, else_layout, result] : llvm::zip_equal(
             then_yield_in_layouts, else_yield_in_layouts, op.getResults())) {
      if (auto vty = dyn_cast<VectorType>(result.getType())) {
        if (!then_layout.has_value()) {
          return op.emitOpError(
                     "expected a vector layout for then yield input ")
                 << out_idx;
        }
        if (!else_layout.has_value()) {
          return op.emitOpError(
                     "expected a vector layout for else yield input ")
                 << out_idx;
        }
        auto compatible_layout = VectorLayout::join(
            then_layout.value(), else_layout.value(), vty.getShape());
        // If no compatible layout is found in layouts for then and else
        // branches, the output layout falls back to a normalized layout which
        // has offsets 0 and the native tiling.
        if (!compatible_layout.has_value()) {
          compatible_layout = VectorLayout(
              then_layout->bitwidth(), {0, 0},
              nativeTiling(then_layout->bitwidth()), ImplicitDim::kNone);
        }
        out_layouts.push_back(compatible_layout);
      } else {
        if (then_layout.has_value()) {
          return op.emitOpError("expected no layout for then yield input ")
                 << out_idx;
        }
        if (else_layout.has_value()) {
          return op.emitOpError("expected no layout for else yield input ")
                 << out_idx;
        }
        out_layouts.push_back(kNoLayout);
      }
      ++out_idx;
    }
    setInLayout(then_yield, out_layouts);
    setInLayout(else_yield, out_layouts);
    setOutLayout(op, out_layouts);
    return success();
  }

  LogicalResult infer(scf::ForOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op.getRegion().hasOneBlock(),
                 "expected one block for scf.for");
    TPU_CHECK_OP(
        op.getNumRegionIterArgs() == op.getNumResults(),
        "expected num_region_iter_args is equal to num_results in scf.for");
    TPU_CHECK_OP(
        op->getNumOperands() == 3 + op.getNumResults(),
        "expected num_operands is equal to 3 + num_results in scf.for");

    auto in_layouts = getLayoutFromOperands(op);
    // Drop the input layouts for lower bound, upper bound. But keep the layout
    // for step because it matches with induction variable in arguments.
    auto arg_layouts = ArrayRef<Layout>(in_layouts).drop_front(2);
    if (assumeLayoutsForBlockArgs(*op.getBody(), arg_layouts).failed() ||
        inferBlock(*op.getBody(), match_yield).failed()) {
      return op.emitOpError(
          "failed to infer layout with initial layouts for body in "
          "scf.for op");
    }
    auto yield_op = op.getBody()->getTerminator();
    auto yield_in_layouts = getLayoutFromOperands(yield_op);

    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    int out_idx = 0;
    bool require_reinfer = false;
    for (auto [in_layout, yield_layout, result] :
         llvm::zip_equal(arg_layouts.drop_front(
                             1),  // Drop the layout for induction variable.
                         yield_in_layouts, op.getResults())) {
      if (auto vty = dyn_cast<VectorType>(result.getType())) {
        if (!in_layout.has_value()) {
          return op.emitOpError("expected a vector layout for input ")
                 << out_idx;
        }
        if (!yield_layout.has_value()) {
          return op.emitOpError("expected a vector layout for yield input ")
                 << out_idx;
        }
        auto compatible_layout = VectorLayout::join(
            in_layout.value(), yield_layout.value(), vty.getShape());
        // If no compatible layout is found in layouts for input and
        // yield, the output layout falls back to a normalized layout which
        // has offsets 0 and the native tiling.
        if (!compatible_layout.has_value()) {
          compatible_layout = VectorLayout(in_layout->bitwidth(), {0, 0},
                                           nativeTiling(in_layout->bitwidth()),
                                           ImplicitDim::kNone);
        }
        if (!require_reinfer &&
            (compatible_layout.value() != in_layout.value() ||
             compatible_layout.value() != yield_layout.value())) {
          require_reinfer = true;
        }
        out_layouts.push_back(compatible_layout);
      } else {
        if (in_layout.has_value()) {
          return op.emitOpError("expected no layout for input ") << out_idx;
        }
        if (yield_layout.has_value()) {
          return op.emitOpError("expected no layout for yield input ")
                 << out_idx;
        }
        out_layouts.push_back(kNoLayout);
      }
      ++out_idx;
    }
    if (require_reinfer) {
      // Force same layouts in input layout but skip the first 3 layouts for
      // lower bound, upper bound and step.
      std::copy(out_layouts.begin(), out_layouts.end(), in_layouts.begin() + 3);

      // Terminator in the loop will carry layouts to the next loop but
      // the loop's block args' layouts are determined by the initial inputs. We
      // need to force the same layouts for all in order to make layouts be
      // consistent across all branches. To ensure that, we need to reprocess
      // layout inference for the entire body with the final consolidated
      // layout.
      clearBlockLayouts(*op.getBody());
      if (assumeLayoutsForBlockArgs(*op.getBody(),
                                    ArrayRef<Layout>(in_layouts).drop_front(2))
              .failed() ||
          inferBlock(*op.getBody(), match_yield).failed()) {
        return op.emitOpError(
            "failed to infer layout with compatible layouts for body in "
            "scf.for op");
      }
    }
    setInLayout(yield_op, out_layouts);
    setLayout(op, in_layouts, out_layouts);
    return success();
  }

  LogicalResult infer(scf::WhileOp op) {
    static LogicalResult (*match_condition)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::ConditionOp>(op), "expected condition terminator");
      return success();
    };
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op.getNumRegions() == 2, "expected two blocks for scf.while");

    SmallVector<Layout, 4> in_layouts = getLayoutFromOperands(op);

    if (assumeLayoutsForBlockArgs(*op.getBeforeBody(), in_layouts).failed() ||
        inferBlock(*op.getBeforeBody(), match_condition).failed()) {
      return op.emitOpError(
          "failed to infer layout with initial layouts for before body in "
          "scf.while op");
    }

    if (assumeLayoutsForBlockArgs(*op.getAfterBody(), in_layouts).failed() ||
        inferBlock(*op.getAfterBody(), match_yield).failed()) {
      return op.emitOpError(
          "failed to infer layout with initial layouts for after body in "
          "scf.while op");
    }

    auto *cond_op = op.getBeforeBody()->getTerminator();
    auto cond_in_layouts = getLayoutFromOperands(cond_op);
    auto *yield_op = op.getAfterBody()->getTerminator();
    auto yield_in_layouts = getLayoutFromOperands(yield_op);

    // Find a compatible layout from condition body and loop body for each
    // reuslt. For example, if we yield offset (*, *) in condition body and
    // offset (*, 0) in loop body, the result offset should be (*, 0).
    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    int out_idx = 0;
    bool require_reinfer = false;
    for (auto [in_layout, cond_layout, yield_layout, result] : llvm::zip_equal(
             in_layouts, ArrayRef<Layout>(cond_in_layouts).drop_front(1),
             yield_in_layouts, op.getResults())) {
      if (auto vty = dyn_cast<VectorType>(result.getType())) {
        if (!in_layout.has_value()) {
          return op.emitOpError("expected a vector layout for whileOp input ")
                 << out_idx;
        }
        if (!cond_layout.has_value()) {
          return op.emitOpError("expected a vector layout for condition input ")
                 << out_idx + 1;  // ConditionOp's first input is 1 bit bool.
        }
        if (!yield_layout.has_value()) {
          return op.emitOpError("expected a vector layout for yield input ")
                 << out_idx;
        }
        auto compatible_layout = VectorLayout::join(
            cond_layout.value(), yield_layout.value(), vty.getShape());
        if (compatible_layout.has_value()) {
          compatible_layout = VectorLayout::join(
              in_layout.value(), compatible_layout.value(), vty.getShape());
        }
        // If no compatible layout is found in layouts for input, condition and
        // yield, the output layout falls back to a normalized layout which
        // has offsets 0 and the native tiling.
        if (!compatible_layout.has_value()) {
          compatible_layout = VectorLayout(in_layout->bitwidth(), {0, 0},
                                           nativeTiling(in_layout->bitwidth()),
                                           ImplicitDim::kNone);
        }
        if (!require_reinfer &&
            (compatible_layout.value() != in_layout.value() ||
             compatible_layout.value() != cond_layout.value() ||
             compatible_layout.value() != yield_layout.value())) {
          require_reinfer = true;
        }
        out_layouts.push_back(compatible_layout);
      } else {
        if (in_layout.has_value()) {
          return op.emitOpError("expected no layout for whileOp input ")
                 << out_idx;
        }
        if (cond_layout.has_value()) {
          return op.emitOpError("expected no layout for condition input ")
                 << out_idx + 1;  // ConditionOp's first input is 1 bit bool.
        }
        if (yield_layout.has_value()) {
          return op.emitOpError("expected no layout for yield input ")
                 << out_idx;
        }
        out_layouts.push_back(kNoLayout);
      }
      ++out_idx;
    }
    if (require_reinfer) {
      clearBlockLayouts(*op.getBeforeBody());
      clearBlockLayouts(*op.getAfterBody());
      // Terminator in the loop will carry layouts to the next loop but
      // the loop's block args' layouts are determined by the initial inputs. We
      // need to force the same layouts for all in order to make layouts be
      // consistent across all branches. To ensure that, we need to reprocess
      // layout inference for the entire body with the final consolidated
      // layout.
      if (assumeLayoutsForBlockArgs(*op.getBeforeBody(), out_layouts)
              .failed() ||
          inferBlock(*op.getBeforeBody(), match_condition).failed()) {
        return op.emitOpError(
            "failed to infer layout with compatible layouts for before body in "
            "scf.while op");
      }
      if (assumeLayoutsForBlockArgs(*op.getAfterBody(), out_layouts).failed() ||
          inferBlock(*op.getAfterBody(), match_yield).failed()) {
        return op.emitOpError(
            "failed to infer layout with compatible layouts for after body in "
            "scf.while op");
      }
    }
    std::copy(out_layouts.begin(), out_layouts.end(),
              cond_in_layouts.begin() + 1);  // Skip the first 1 bit bool.
    setInLayout(cond_op, cond_in_layouts);
    setInLayout(yield_op, out_layouts);
    setLayout(op, out_layouts, out_layouts);
    return success();
  }

  // TODO(b/347016737): deprecate the static rotate.
  LogicalResult infer(tpu::RotateOp op) {
    auto bitwidth = op.getType().getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Rotate with non-32-bit data");
    }
    if (op.getType().getRank() < 2) {
      NYI("Unsupported 1D shape");
    }
    auto layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                               ImplicitDim::kNone);
    setLayout(op, layout, layout);
    return success();
  }

  LogicalResult infer(tpu::DynamicRotateOp op) {
    auto bitwidth = op.getType().getElementTypeBitWidth();
    // TODO(b/347067057): Support dynamic rotate with packed dtype.
    if (bitwidth != 32) {
      NYI("Rotate with non-32-bit data");
    }
    if (op.getType().getRank() < 2) {
      NYI("Unsupported 1D shape");
    }
    auto layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                               ImplicitDim::kNone);
    setLayout(op, {layout, kNoLayout}, layout);
    return success();
  }

  LogicalResult infer(tpu::ConcatenateOp op) {
    TPU_CHECK_OP(!op.getSources().empty(),
                 "Need at least one vector to concatenate");
    auto res_rank = op.getType().getRank();
    auto dimension = op.getDimension();
    TPU_CHECK_OP(0 <= dimension && dimension < res_rank,
                 "Expect a valid concatenate dimension");
    if (res_rank == 1) {
      NYI("Support concatenation with 1D vectors");
    }
    auto res_ty = op.getResult().getType();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();
    auto layout = getLayout(op.getSources().front());
    // When concatenating vectors with replicated offsets, we want to reset the
    // replicated offset to zero. Because we are not sure if the replicated
    // value from each vector are same.
    layout = VectorLayout(
        layout->bitwidth(),
        {layout->offsets()[0].value_or(0), layout->offsets()[1].value_or(0)},
        layout->tiling(), layout->implicit_dim());
    if (dimension >= res_rank - 2) {
      layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                            ImplicitDim::kNone);
    }
    SmallVector<Layout> in_layouts(op->getNumOperands(), layout);
    setLayout(op, in_layouts, layout);
    return success();
  }

  LogicalResult infer(tpu::LoadOp op) {
    auto res_ty = op.getResult().getType();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();

    // We expect the result is already a native-sized vreg.
    TPU_CHECK_OP(bitwidth == 32 && res_ty.getShape()[0] == target_shape_[0] &&
                     res_ty.getShape()[1] == target_shape_[1],
                 "Only 32-bit loads supported");
    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    auto out_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                   ImplicitDim::kNone);
    setLayout(op, in_layout, out_layout);
    return success();
  }

  LogicalResult infer(tpu::StridedLoadOp op) {
    auto vty = op.getResult().getType();
    int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Strided load with non 32-bit data");
    }
    if (vty.getRank() < 2) {
      NYI("Strided load with 1D vector");
    }
    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    setLayout(op, in_layout,
              VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                           ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(tpu::StridedStoreOp op) {
    auto vty = op.getValueToStore().getType();
    int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Strided store with non 32-bit data");
    }
    if (vty.getRank() < 2) {
      NYI("Strided store with 1D vector");
    }
    auto store_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                     ImplicitDim::kNone);
    SmallVector<Layout, 5> in_layout{op->getNumOperands(), kNoLayout};
    in_layout[0] = store_layout;
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(tpu::MatmulOp op) { return inferMatmul(op); }

  LogicalResult infer(tpu::StoreOp op) {
    auto store_ty = op.getValueToStore().getType();
    int8_t bitwidth = store_ty.getElementTypeBitWidth();

    // We expect the value to store is already a native-sized vreg.
    TPU_CHECK_OP(bitwidth == 32 && store_ty.getShape()[0] == target_shape_[0] &&
                     store_ty.getShape()[1] == target_shape_[1],
                 "Only 32-bit stores supported");
    auto store_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                     ImplicitDim::kNone);
    SmallVector<Layout, 5> in_layout{store_layout};
    in_layout.insert(in_layout.end(), op.getIndices().size() + 1, kNoLayout);
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(tpu::EraseLayoutOp op) {
    setLayout(op, kNoLayout, kNoLayout);
    return success();
  }

  LogicalResult infer(tpu::GatherOp op) {
    auto src_layout = getLayout(op.getSource());
    setLayout(op, src_layout, src_layout);
    return success();
  }

  LogicalResult infer(tpu::BitcastOp op) {
    auto src_layout = getLayout(op.getInput());
    LayoutOffsets src_offsets = src_layout->offsets();
    if (src_offsets[0].value_or(0) || src_offsets[1].value_or(0)) {
      NYI("unsupported bitcast with offsets");
    }
    if (src_layout->implicit_dim() != ImplicitDim::kNone) {
      NYI("unsupported bitcast with an implicit dim");
    }
    // Check if input and output have same bit size.
    auto in_ty = dyn_cast<VectorType>(op.getInput().getType());
    auto out_ty = dyn_cast<VectorType>(op.getOutput().getType());
    auto in_bitwidth = in_ty.getElementTypeBitWidth();
    auto out_bitwidth = out_ty.getElementTypeBitWidth();
    TPU_CHECK_OP(in_ty && out_ty && in_ty.getRank() == out_ty.getRank(),
                 "Input and output have different rank");
    if (out_ty.getRank() < 2) {
      NYI("Support bitcast with 1D vector");
    }
    for (int i = 0; i < in_ty.getRank(); ++i) {
      auto in_dim = in_ty.getDimSize(i);
      auto out_dim = out_ty.getDimSize(i);

      // The sublane dimension is scaled down by the ratio of input element
      // bitwidth to output element bitwidth when bitcasting. For example,
      // bitcasting a vector<16x128xbf16> to a vector<8x128xi32> packs every 2
      // rows in the bf16 vector into 1 row in the i32 vector. This means the
      // bit representation of one i32 element vector[i,j] is equal to
      // concatenating bf16 elements vector[2*i+1,j] and vector[2*i,j].
      if (i == in_ty.getRank() - 2) {
        in_dim *= in_bitwidth;
        out_dim *= out_bitwidth;
      }
      TPU_CHECK_OP(in_dim == out_dim,
                   "Input and output have incompatible shape");
    }
    setLayout(op,
              VectorLayout(in_bitwidth, src_offsets, nativeTiling(in_bitwidth),
                           ImplicitDim::kNone),
              VectorLayout(out_bitwidth, src_offsets,
                           nativeTiling(out_bitwidth), ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(tpu::RepeatOp op) {
    auto src_layout = getLayout(op.getSource());
    setLayout(op, src_layout, src_layout);
    return success();
  }

  LogicalResult infer(tpu::TraceOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<tpu::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 0, "expected no operands");
    TPU_CHECK_OP(op->getNumResults() == 0, "results unsupported");
    return inferBlock(*op.getBody(), match_yield);
  }

  LogicalResult infer(tpu::RegionOp op) {
    static LogicalResult (*match_region)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<tpu::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 0, "expected no operands");
    TPU_CHECK_OP(op->getNumResults() == 0, "results unsupported");
    return inferBlock((*op).getRegion(0).getBlocks().front(), match_region);
  }

  LogicalResult infer(tpu::IotaOp op) {
    auto ty = op.getResult().getType();
    TPU_CHECK_OP(ty.getElementType().isSignlessInteger(32),
                 "Only 32-bit integer iota supported");
    TPU_CHECK_OP(ty.getRank() >= 2, "iota rank below 2D unsupported");
    LayoutOffsets offsets = {0, 0};
    if (op.getDimension() == ty.getRank() - 1) {
      offsets[0] = std::nullopt;
    }
    if (op.getDimension() == ty.getRank() - 2) {
      offsets[1] = std::nullopt;
    }
    setOutLayout(op, VectorLayout(kNativeBitwidth, offsets, default_tiling_,
                                  ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(vector::BroadcastOp op) {
    auto some_src_ty = op.getSourceType();
    auto res_ty = op.getResultVectorType();
    TPU_CHECK_OP(res_ty.getRank() > 0, "rank 0 vectors unsupported");
    if (some_src_ty.isSignlessIntOrIndexOrFloat()) {
      auto bitwidth = some_src_ty.getIntOrFloatBitWidth();
      // TODO(b/320725357): We need a better design for mask layout. For now, we
      // always set layout bitwidth of Vmask to 32bit.
      if (bitwidth == 1) {
        bitwidth = kNativeBitwidth;
      }
      if (res_ty.getRank() == 1) {
        // We use a full vreg tile, because only then its layout can be changed
        // for free.
        setLayout(
            op, kNoLayout,
            VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                         nativeTiling(bitwidth), ImplicitDim::kSecondMinor));
      } else {  // rank >= 2  // NOLINT(readability-else-after-return)
        setLayout(op, kNoLayout,
                  VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                               nativeTiling(bitwidth), ImplicitDim::kNone));
      }
      return success();
    }
    if (auto src_ty = dyn_cast<VectorType>(some_src_ty)) {
      TPU_CHECK_OP(src_ty.getRank() >= 2, "source rank below 2D unsupported");
      TPU_CHECK_OP(res_ty.getRank() >= 2, "result rank below 2D unsupported");
      auto some_layout = getLayout(op.getSource());
      TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
      auto &layout = *some_layout;
      // Since we can only do sublane broadcasts in the (8, 128) tiling, we
      // should always use that when sublane broadcasting is required.
      if (*(src_ty.getShape().end() - 2) != *(res_ty.getShape().end() - 2)) {
        if (layout.bitwidth() != kNativeBitwidth) {
          NYI("Only 32-bit broadcasts supported");
        }
        LayoutOffsets offsets = layout.offsets();
        // At the moment relayout can only produce replicated sublanes when
        // converting to (8, 128) if the input was in (1, 128) tiling
        if (layout.tiling()[0] == 1) {
          offsets[0] = std::nullopt;
        }
        layout = VectorLayout(layout.bitwidth(), offsets, default_tiling_,
                              layout.implicit_dim());
      }
      if (layout.implicit_dim() != ImplicitDim::kNone) {
        VectorLayout layout_2d(layout.bitwidth(), layout.offsets(),
                               layout.tiling(), ImplicitDim::kNone);
        if (layout_2d.equivalentTo(layout, src_ty.getShape(), target_shape_)) {
          // TODO(b/342237796): Stop preferring 2D layouts (if given the choice)
          // and defer the work, if any, to relayout.
          layout = layout_2d;
        }
      }
      auto src_tiled_shape = src_ty.getShape().take_back(2);
      auto dst_tiled_shape = res_ty.getShape().take_back(2);
      LayoutOffsets offsets = layout.offsets();
      for (int i = 0; i < 2; ++i) {
        if (src_tiled_shape[i] != dst_tiled_shape[i]) {
          offsets[i] = std::nullopt;
        }
      }
      setLayout(op, layout,
                VectorLayout(layout.bitwidth(), offsets, layout.tiling(),
                             layout.implicit_dim()));
      return success();
    }
    op.emitOpError("unsupported broadcast source type");
    return failure();
  }

  LogicalResult infer(vector::ContractionOp op) {
    // TODO(apaszke): Support layout here, at least on batch dimensions.
    TPU_CHECK_OP(op.getKind() == vector::CombiningKind::ADD,
                 "Only ADD supported");
    auto ctx = op.getContext();
    const auto matmul_iterator_types = mlir::ArrayAttr::get(
        ctx,
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
    TPU_CHECK_OP(op.getIteratorTypes() == matmul_iterator_types,
                 "Not a matmul");
    const auto matmul_indexing_maps = mlir::ArrayAttr::get(
        ctx,
        {AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(2, ctx), getAffineDimExpr(1, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)},
             ctx))});
    const auto matmul_indexing_maps_transposed = mlir::ArrayAttr::get(
        ctx,
        {AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)},
             ctx))});
    TPU_CHECK_OP(op.getIndexingMaps() == matmul_indexing_maps ||
                     op.getIndexingMaps() == matmul_indexing_maps_transposed,
                 "Not a matmul");
    return inferMatmul(op);
  }

  LogicalResult infer(vector::ExtractOp op) {
    TPU_CHECK_OP(!op.hasDynamicPosition(), "dynamic indices not supported");
    TPU_CHECK_OP(
        op.getSourceVectorType().getElementTypeBitWidth() == kNativeBitwidth,
        "Only 32-bit types supported");
    auto layout = getLayout(op.getVector());
    TPU_CHECK_OP(layout.has_value(), "missing vector layout");
    if (VectorType res_vty = dyn_cast<VectorType>(op.getResult().getType());
        res_vty != nullptr) {
      if (res_vty.getRank() == 1 &&
          layout->implicit_dim() == ImplicitDim::kNone) {
        const int64_t second_minor_idx = op.getStaticPosition().back();
        const LayoutOffset second_minor_offset = layout->offsets()[0];
        const LayoutOffset res_second_minor_offset =
            second_minor_offset.has_value()
                ? (*second_minor_offset + second_minor_idx) %
                      layout->vregSlice(target_shape_)[0]
                : LayoutOffset();
        setLayout(op, layout,
                  VectorLayout(layout->bitwidth(),
                               {res_second_minor_offset, layout->offsets()[1]},
                               layout->tiling(), ImplicitDim::kSecondMinor));
      } else {
        TPU_CHECK_OP(layout->layout_rank() <= res_vty.getRank(),
                     "Internal error: Layout has too many dimensions for "
                     "vector type (invalid vector.extract?)")
        setLayout(op, layout, layout);
      }
    } else {
      setLayout(op,
                VectorLayout(kNativeBitwidth, {0, 0}, layout->tiling(),
                             layout->implicit_dim()),
                kNoLayout);
    }
    return success();
  }

  LogicalResult infer(vector::LoadOp op) {
    auto src_ty = op.getMemRefType();
    auto res_ty = op.getVectorType();
    TPU_CHECK_OP(src_ty.getRank() == res_ty.getRank(),
                 "memref and vector rank mismatch");
    int64_t rank = res_ty.getRank();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();
    if (kNativeBitwidth % bitwidth != 0) {
      return op.emitOpError("Unsupported bitwidth");
    }
    const int packing = kNativeBitwidth / bitwidth;
    auto maybe_tiling =
        verifyMemoryTiling(op, getMemRefLayout(op.getBase()).getTiles(),
                           src_ty.getRank(), src_ty.getElementTypeBitWidth());
    if (!maybe_tiling) {
      return failure();
    }
    auto tiling = *maybe_tiling;

    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    CHECK_EQ(op->getNumOperands(), op.getIndices().size() + 1);
    SmallVector<int64_t, 2> tile_offsets;  // indices % tiling
    for (int i = 0; i < tiling.size(); ++i) {
      int dim = rank - tiling.size() + i;
      Value tiled_index = op.getIndices()[dim];
      if (auto cst_op = tiled_index.getDefiningOp<arith::ConstantOp>()) {
        tile_offsets.push_back(cast<IntegerAttr>(cst_op.getValue()).getInt() %
                               tiling[i]);
      } else {
        if (failed(verifyDivisibleIndex(tiled_index, tiling[i], dim, op))) {
          return failure();
        }
        tile_offsets.push_back(0);
      }
    }

    if (rank == 0) {
      op.emitOpError("rank 0 vectors unsupported");
      return failure();
    }
    if (rank == 1) {
      TPU_CHECK_OP(tiling.size() == 1, "Expected 1D tiling in 1D loads");
      const int64_t lane_tiling = packing * target_shape_[1];
      auto tile = tiling.front();
      TPU_CHECK_OP(tile % lane_tiling == 0, "Unsupported tiling for 1D load");
      CHECK_EQ(tile_offsets.size(), 1);
      // TODO(apaszke): We could generate replicated loads for short values.
      setLayout(op, in_layout,
                VectorLayout(bitwidth, {0, tile_offsets[0] % lane_tiling},
                             {1, lane_tiling}, ImplicitDim::kSecondMinor));
    } else {  // rank >= 2
      TPU_CHECK_OP(tiling.size() == 2, "Expected 2D tiling in 2D+ loads");
      CHECK_EQ(tile_offsets.size(), 2);
      std::array<std::optional<int64_t>, 2> offsets;
      const auto tile_src_shape = src_ty.getShape().take_back(2);
      const auto tile_res_shape = res_ty.getShape().take_back(2);
      const int64_t num_sublanes = tile_res_shape[0];
      // For now, we focus on tilings that span full sublanes.
      TPU_CHECK_OP(tiling[1] == target_shape_[1],
                   "Unsupported tiling for 2d load");
      // We can load starting from any row if the source has few columns,
      // because the tiling structure degenerates to regular layout there.
      // There is also no extra need for alignment if we load a single sublane.
      // TODO(apaszke): Also no need to align if we don't exceed the base chunk!
      if (bitwidth == 32 &&
          (tile_src_shape[1] <= target_shape_[1] || num_sublanes == 1)) {
        offsets[0] = 0;
      } else {
        offsets[0] = tile_offsets[0];
      }
      offsets[1] = tile_offsets[1];
      std::array<int64_t, 2> layout_tiling{tiling[0], tiling[1]};
      if (num_sublanes == 1 && bitwidth == 32 &&
          tiling[1] == target_shape_[1] &&
          tile_res_shape[1] > target_shape_[1]) {
        // We can strided load sublanes if we're loading a single sublane for
        // multiple times. Enabling this helps load one entire row from memref
        // more efficiently.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, offsets, {1, layout_tiling[1]},
                               ImplicitDim::kNone));
      } else if (num_sublanes == 1 && bitwidth == 32 &&
                 tiling == target_shape_) {
        // We can use replicated loads if we're only loading a single sublane.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, {std::nullopt, offsets[1]},
                               layout_tiling, ImplicitDim::kNone));
      } else {
        setLayout(
            op, in_layout,
            VectorLayout(bitwidth, offsets, layout_tiling, ImplicitDim::kNone));
      }
    }
    return success();
  }

  LogicalResult infer(vector::ExtractStridedSliceOp op) {
    auto input_layout = getLayout(op.getVector());
    TPU_CHECK_OP(input_layout, "missing vector layout");
    auto offsets_attr = op.getOffsets().getValue();
    auto strides_attr = op.getStrides().getValue();
    auto offsets = llvm::map_to_vector(offsets_attr, [](auto attr) {
      return cast<IntegerAttr>(attr).getInt();
    });
    input_layout->insertImplicit<int64_t>(offsets, 0);
    auto vreg_slice = input_layout->vregSlice(target_shape_);
    LayoutOffsets new_layout_offsets;
    if (input_layout->offsets()[0].has_value()) {
      new_layout_offsets[0] =
          (*(offsets.end() - 2) + *input_layout->offsets()[0]) % vreg_slice[0];
    }
    if (input_layout->offsets()[1].has_value()) {
      new_layout_offsets[1] =
          (*(offsets.end() - 1) + *input_layout->offsets()[1]) % vreg_slice[1];
    }
    for (auto stride : strides_attr) {
      TPU_CHECK_OP(stride.cast<IntegerAttr>().getInt() == 1,
                   "Only trivial strides supported.");
    }

    setLayout(
        op, input_layout,
        VectorLayout(input_layout->bitwidth(), new_layout_offsets,
                     input_layout->tiling(), input_layout->implicit_dim()));
    return success();
  }

  LogicalResult infer(vector::MultiDimReductionOp op) {
    auto src_ty = op.getSourceVectorType();
    auto dst_ty = dyn_cast<VectorType>(op.getDestType());
    TPU_CHECK_OP(dst_ty, "only reductions with vector results supported");
    SmallVector<int64_t> dims;
    dims.reserve(op.getReductionDims().size());
    for (Attribute dim_attr : op.getReductionDims()) {
      dims.push_back(cast<IntegerAttr>(dim_attr).getInt());
    }
    int64_t src_rank = src_ty.getRank();
    auto acc_layout = getLayout(op.getAcc());
    TPU_CHECK_OP(is_fully_replicated(acc_layout),
                 "only constant accumulators supported");
    TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == kNativeBitwidth,
                 "only 32-bit reductions supported");
    auto some_src_layout = getLayout(op.getSource());
    TPU_CHECK_OP(some_src_layout, "missing vector layout");
    auto &src_layout = *some_src_layout;
    std::array<bool, 2> reduces;
    switch (src_layout.implicit_dim()) {
      case VectorLayout::ImplicitDim::kNone:
        reduces = {
            std::find(dims.begin(), dims.end(), src_rank - 2) != dims.end(),
            std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end()};
        break;
      case VectorLayout::ImplicitDim::kSecondMinor:
        reduces = {false, std::find(dims.begin(), dims.end(), src_rank - 1) !=
                              dims.end()};
        break;
      case VectorLayout::ImplicitDim::kMinor:
        reduces = {
            std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end(),
            false};
        break;
    }
    if ((reduces[0] || reduces[1]) &&
        !src_layout.hasNativeTiling(target_shape_)) {
      src_layout = VectorLayout(kNativeBitwidth, src_layout.offsets(),
                                default_tiling_, src_layout.implicit_dim());
    }
    LayoutOffsets out_offsets = src_layout.offsets();
    for (int i = 0; i < out_offsets.size(); ++i) {
      if (reduces[i]) {
        out_offsets[i] = std::nullopt;
      }
    }
    ImplicitDim out_implicit_dim = src_layout.implicit_dim();
    if ((reduces[0] && reduces[1]) ||
        (src_layout.implicit_dim() != ImplicitDim::kNone &&
         (reduces[0] || reduces[1]))) {
      TPU_CHECK_OP(
          dst_ty.getRank() > 0 && *(dst_ty.getShape().end() - 1) == 1,
          "Not implemented: reductions over both trailing dimensions are only "
          "supported when the resulting value has a trailing axis of size 1");
      out_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
    } else if (reduces[0]) {
      out_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
    } else if (reduces[1]) {
      out_implicit_dim = VectorLayout::ImplicitDim::kMinor;
    }
    setLayout(op, {src_layout, acc_layout},
              VectorLayout(src_layout.bitwidth(), out_offsets,
                           src_layout.tiling(), out_implicit_dim));
    return success();
  }

  LogicalResult infer(vector::ShapeCastOp op) {
    auto src_ty = op.getSourceVectorType();
    auto src_shape = src_ty.getShape();
    int64_t src_rank = src_ty.getRank();
    auto res_ty = op.getResultVectorType();
    auto res_shape = res_ty.getShape();
    int64_t res_rank = res_ty.getRank();
    auto some_src_layout = getLayout(op.getSource());
    TPU_CHECK_OP(some_src_layout, "missing vector layout");
    auto layout = *some_src_layout;
    const unsigned bitwidth = src_ty.getElementTypeBitWidth();
    const std::array<int64_t, 2> vreg_slice = layout.vregSlice(target_shape_);
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      // Nothing changes in the last two dims.
      if (res_rank >= 2 && src_shape.take_back(2) == res_shape.take_back(2)) {
        setLayout(op, layout, layout);
        return success();
      }
      // Sublane (un)tiling.
      if (res_rank >= 2 && *(src_shape.end() - 1) == *(res_shape.end() - 1) &&
          *(src_shape.end() - 2) % vreg_slice[0] == 0 &&
          *(res_shape.end() - 2) % vreg_slice[0] == 0) {
        // TODO(b/343808585): We shouldn't force second minor offset to 0 when
        //                    unfolding, it's still a no-op, but we need to add
        //                    support in apply-vector-layout.
        layout = VectorLayout(layout.bitwidth(), {0, layout.offsets()[1]},
                              layout.tiling(), layout.implicit_dim());
        setLayout(op, layout, layout);
        return success();
      }
      const auto native_tiling = nativeTiling(bitwidth);
      // Lane (un)tiling.
      if (src_ty.getDimSize(src_ty.getRank() - 1) !=
              res_shape[res_shape.size() - 1] &&
          src_ty.getDimSize(src_ty.getRank() - 1) % layout.tiling()[1] == 0 &&
          res_shape[res_shape.size() - 1] % layout.tiling()[1] == 0) {
        const int packing = kNativeBitwidth / bitwidth;
        const auto elements_per_vreg = native_tiling[0] * native_tiling[1];
        // When we shapecast from input shape
        // (..., m * target_shape_[1] * packing) to output shape
        // (..., target_shape_[1]), the reshape becomes no-op when input is
        // densely packed with tiling (1, target_shape_[1] * packing) and output
        // has the native tiling.
        if (*(res_shape.end() - 1) == target_shape_[1] &&
            *(res_shape.end() - 2) % native_tiling[0] == 0 &&
            *(src_shape.end() - 1) % elements_per_vreg == 0) {
          // Inferring in_layout to have tiling (1, 128 * packing) triggers any
          // necessary relayout before shapecast.
          setLayout(
              op,
              VectorLayout(layout.bitwidth(), {0, 0},
                           {1, target_shape_[1] * packing}, ImplicitDim::kNone),
              VectorLayout(layout.bitwidth(), {0, 0}, native_tiling,
                           ImplicitDim::kNone));
          return success();
        }

        // When we shapecast from input shape (..., target_shape_[1]) to output
        // shape (..., m * target_shape_[1] * packing), the reshape becomes
        // no-op when input has the native tiling and output is densely packed
        // with tiling (1, target_shape_[1] * packing).
        if (*(src_shape.end() - 1) == target_shape_[1] &&
            *(src_shape.end() - 2) % native_tiling[0] == 0 &&
            *(res_shape.end() - 1) % elements_per_vreg == 0) {
          setLayout(op,
                    VectorLayout(layout.bitwidth(), {0, 0}, native_tiling,
                                 ImplicitDim::kNone),
                    VectorLayout(layout.bitwidth(), {0, 0},
                                 {1, target_shape_[1] * packing},
                                 ImplicitDim::kNone));
          return success();
        }

        // TODO(b/299253805): support shapecast along lane for other cases.
        op.emitOpError("unsupported shape cast");
        return failure();
      }
      if (layout.tiling() != native_tiling) {
        layout = VectorLayout(bitwidth, layout.offsets(), native_tiling,
                              layout.implicit_dim());
      }
      TPU_CHECK_OP(src_ty.getRank() >= 2,
                   "expected 2D+ operand with 2D layout");
      ArrayRef<int64_t> layout_shape = src_ty.getShape().take_back(2);
      if (res_ty.getRank() >= 2) {
        // Squeeze out the sublane dim.
        if (layout_shape[0] == 1 &&
            res_shape.back() == src_shape.back()) {
          setLayout(op, layout,
                    VectorLayout(bitwidth, layout.offsets(), layout.tiling(),
                                 ImplicitDim::kSecondMinor));
          return success();
        }
        // Insert a singleton lane dimension. The old lane dimension ends up
        // in the sublane dimension. Other axes can be reshaped arbitrarily.
        if (src_ty.getElementTypeBitWidth() == kNativeBitwidth &&
            src_shape.back() == res_shape[res_shape.size() - 2] &&
            res_shape.back() == 1) {
          setLayout(op, layout,
                    VectorLayout(kNativeBitwidth, {0, std::nullopt},
                                 default_tiling_, ImplicitDim::kNone));
          return success();
        }
      } else if (res_ty.getRank() == 1) {
        // All dimensions have been folded into a single one

        // Squeeze all but minor dimension
        if (res_ty.getShape().back() == layout_shape[1]) {
          // The condition implies that everything apart from the minor
          // dimension is 1 in the source.
          setLayout(op, layout,
                    VectorLayout(bitwidth, layout.offsets(), layout.tiling(),
                                 ImplicitDim::kSecondMinor));
          return success();
        }
        // Squeeze all but second minor dimension
        if (res_ty.getShape().back() == layout_shape[0]) {
          // The condition implies that everything apart from the second minor
          // dimension is 1 in the source
          setLayout(op, layout,
                    VectorLayout(kNativeBitwidth, layout.offsets(),
                                 layout.tiling(), ImplicitDim::kMinor));
          return success();
        }
        // TODO(b/340625465): Add case where layout_shape is (1, 1) and we fold
        //                    batch dimensions once we support 0-D layouts.
      }
    } else {
      // Nothing changes in the last dim.
      if (res_ty.getRank() >= 1 && src_shape.back() == res_shape.back()) {
        setLayout(op, layout, layout);
        return success();
      }
      // Insert a singleton innermost dim.
      if (res_ty.getRank() == src_ty.getRank() + 1 &&
          src_ty.getDimSize(src_rank - 1) == res_ty.getDimSize(res_rank - 2) &&
          res_ty.getDimSize(res_rank - 1) == 1) {
        if (layout.implicit_dim() == ImplicitDim::kMinor) {
          setLayout(op, layout,
                    VectorLayout(bitwidth, layout.offsets(), layout.tiling(),
                                 ImplicitDim::kNone));
        } else {
          TPU_CHECK_OP(bitwidth == kNativeBitwidth,
                       "Insertion of minor dim that is not a no-op only "
                       "supported for 32-bit types");
          TPU_CHECK_OP(layout.implicit_dim() == ImplicitDim::kSecondMinor,
                       "unexpected implicit dim value");
          setLayout(op, layout,
                    VectorLayout(bitwidth, {0, std::nullopt}, default_tiling_,
                                 ImplicitDim::kNone));
        }
        return success();
      }
    }
    op.emitOpError("unsupported shape cast");
    return failure();
  }

  LogicalResult infer(vector::StoreOp op) {
    auto ref_ty = op.getMemRefType();
    auto store_ty = op.getValueToStore().getType();
    TPU_CHECK_OP(ref_ty.getRank() == store_ty.getRank(),
                 "memref and vector rank mismatch");
    const Layout maybe_src_layout = getLayout(op.getValueToStore());
    TPU_CHECK_OP(maybe_src_layout.has_value(), "missing vector layout");
    const VectorLayout &src_layout = *maybe_src_layout;
    int64_t rank = ref_ty.getRank();
    int8_t bitwidth = store_ty.getElementTypeBitWidth();
    if (kNativeBitwidth % bitwidth != 0) {
      return op.emitOpError("Unsupported bitwidth");
    }
    const int packing = kNativeBitwidth / bitwidth;
    auto maybe_tiling =
        verifyMemoryTiling(op, getMemRefLayout(op.getBase()).getTiles(),
                           ref_ty.getRank(), ref_ty.getElementTypeBitWidth());
    if (!maybe_tiling) {
      return failure();
    }
    auto tiling = *maybe_tiling;

    SmallVector<int64_t, 2> tile_offsets;  // indices % tiling
    for (int i = 0; i < tiling.size(); ++i) {
      int dim = rank - tiling.size() + i;
      Value tiled_index = op.getIndices()[dim];
      if (auto cst_op = tiled_index.getDefiningOp<arith::ConstantOp>()) {
        tile_offsets.push_back(cast<IntegerAttr>(cst_op.getValue()).getInt() %
                               tiling[i]);
      } else {
        if (failed(verifyDivisibleIndex(tiled_index, tiling[i], dim, op))) {
          return failure();
        }
        tile_offsets.push_back(0);
      }
    }

    Layout store_layout;
    if (rank == 0) {
      op.emitOpError("rank 0 vectors unsupported");
      return failure();
    }
    if (rank == 1) {
      TPU_CHECK_OP(tiling.size() == 1, "Expected 1D tiling in 1D store");
      const int64_t lane_tiling = packing * target_shape_[1];
      auto tile = tiling.front();
      TPU_CHECK_OP(tile % lane_tiling == 0,
                   "Unsupported 1D tiling for 1D store");
      CHECK_EQ(tile_offsets.size(), 1);
      store_layout =
          VectorLayout(bitwidth, {0, tile_offsets[0] % lane_tiling},
                       {1, lane_tiling}, ImplicitDim::kSecondMinor);
    } else {  // rank >= 2  // NOLINT(readability-else-after-return)
      TPU_CHECK_OP(tiling.size() == 2, "Expected 2D tiling in 2D+ store");
      CHECK_EQ(tile_offsets.size(), 2);
      const auto tile_ref_shape = ref_ty.getShape().take_back(2);
      const auto tile_store_shape = store_ty.getShape().take_back(2);
      const int64_t num_sublanes = tile_store_shape[0];
      // For now, we focus on tilings that span full sublanes.
      TPU_CHECK_OP(tiling[1] == target_shape_[1],
                   "Unsupported tiling for 2d store");

      if (num_sublanes == 1 && bitwidth == 32 &&
          tiling[1] == target_shape_[1] &&
          tile_store_shape[1] > target_shape_[1]) {
        // We can strided store sublanes if we're storing a single sublane for
        // multiple times. Enabling this helps store one entire row to memref
        // more efficiently.
        store_layout = VectorLayout(store_ty.getElementTypeBitWidth(),
                                    {0, tile_offsets[1]}, {1, tiling[1]},
                                    ImplicitDim::kNone);
      } else {
        // For now, here we only support storing from vreg tiling that matches
        // memory tiling. Assume that's the tiling we infer, going forward.
        const std::array<int64_t, 2> vreg_slice = VectorLayout::vregSlice(
            bitwidth, {tiling[0], tiling[1]}, target_shape_);

        // Offsets don't always have to match exactly.
        // TODO(tlongeri): Sometimes, even if we have the freedom to pick any
        //                 offset, it may be worth it to shift to reduce the
        //                 number of vregs we have to store.
        std::array<std::optional<int64_t>, 2> offsets;
        // TODO(tlongeri): Support replicated offsets in store. For now we just
        //                 set them to match memory.
        offsets[0] =
            src_layout.offsets()[0].value_or(tile_offsets[0]) % vreg_slice[0];
        offsets[1] =
            src_layout.offsets()[1].value_or(tile_offsets[1]) % vreg_slice[1];
        // Sublane offsets don't necessarily have to match. However, we
        // currently only have support for doing one contiguous store per vreg.
        if (bitwidth == 32 &&
            (tile_ref_shape[1] <= tiling[1] || num_sublanes == 1)) {
          // We can store starting from any row if the source has few columns,
          // because the tiling structure degenerates to regular layout there.
          // There is also no extra need for alignment if we store a single
          // sublane.
          // TODO(apaszke): Also no need to align if we don't exceed the base
          //                chunk!
        } else {
          // If we have to shift sublanes, might as well pick the smallest
          // allowable offset since it always results in the smallest padding
          // and number of vregs.
          // TODO(tlongeri): Note that in TPU gens where we care about rotate
          //                 amount, this may not be optimal.
          offsets[0] = tile_offsets[0];
        }
        // Lane offsets *within* tile must always match, otherwise sublanes will
        // not be congruent (assuming tiles are one sublane horizontally, which
        // is true of the tilings we support).
        if (*offsets[1] % tiling[1] != tile_offsets[1]) {
          // Similar to sublanes, if we have to shift lanes, then might as well
          // pick the smallest allowable offset since it always results in the
          // smallest padding and number of vregs.
          offsets[1] = tile_offsets[1];
        }
        store_layout = VectorLayout(store_ty.getElementTypeBitWidth(), offsets,
                                    {tiling[0], tiling[1]}, ImplicitDim::kNone);
      }
    }
    SmallVector<Layout, 5> in_layout{store_layout};
    in_layout.insert(in_layout.end(), op.getIndices().size() + 1, kNoLayout);
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(vector::TransposeOp op) {
    auto permutation = op.getPermutation();
    TPU_CHECK_OP(permutation.size() > 1,
                 "Vector and scalar transpose should be a no-op and removed");

    auto some_layout = getLayout(op.getVector());
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    auto src_ty = op.getSourceVectorType();
    TPU_CHECK_OP(permutation.size() == src_ty.getRank(),
                 "Transpose permutation has incorrect rank");
    for (auto dim : permutation.drop_back(2)) {
      TPU_CHECK_OP(dim < src_ty.getRank() - 2,
                   "Unsupported transpose permutation - minor dims into major");
    }
    for (auto dim : permutation.take_back(2)) {
      TPU_CHECK_OP(dim >= src_ty.getRank() - 2,
                   "Unsupported transpose permutation - major dims into minor");
    }
    Layout required_layout = some_layout;
    // Require native tiling if we're going to use the XLU.
    if (permutation[permutation.size() - 1] == permutation.size() - 2) {
      auto native_tiling = nativeTiling(layout.bitwidth());
      required_layout = VectorLayout(layout.bitwidth(), LayoutOffsets{0, 0},
                                     native_tiling, ImplicitDim::kNone);
    }
    setLayout(op, required_layout, required_layout);
    return success();
  }

  LogicalResult inferExt(Operation *op) {
    TPU_CHECK_OP(op->getNumOperands() == 1, "expect 1 operand");
    TPU_CHECK_OP(op->getNumResults() == 1, "expect 1 result");
    auto src_ty = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op->getResult(0).getType());
    auto some_layout = getLayout(op->getOperand(0));
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    if (dyn_cast<arith::ExtFOp>(op)) {
      TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == 16 &&
                       dst_ty.getElementTypeBitWidth() == 32,
                   "Only 16-bit to 32-bit extensions supported");
    } else {
      TPU_CHECK_OP(dst_ty.getElementTypeBitWidth() == 32,
                   "Only extensions to 32-bit supported");
    }
    auto &layout = *some_layout;
    // TODO(apaszke): Support native packed layouts here.
    Layout src_layout;
    Layout dst_layout;
    // All layouts that subdivide the rows of the default tiling evenly
    // can be handled uniformly with the default case, by preserving the
    // tiling through the op.
    if (default_tiling_[0] % layout.tiling()[0] == 0 &&
        default_tiling_[1] == layout.tiling()[1]) {
      src_layout = layout;
      dst_layout = VectorLayout(32, layout.offsets(), src_layout->tiling(),
                                layout.implicit_dim());
    } else if (layout.tiling() ==
               nativeTiling(src_ty.getElementTypeBitWidth())) {
      // If the source is already in native tiling, we can unpack it directly.
      src_layout = layout;
      dst_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                layout.implicit_dim());
    } else {
      // TODO(b/335863273): we should also reduce offsets.
      src_layout = VectorLayout(layout.bitwidth(), layout.offsets(),
                                default_tiling_, layout.implicit_dim());
      dst_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                layout.implicit_dim());
    }
    setLayout(op, src_layout, dst_layout);
    return success();
  }

  LogicalResult inferTrunc(Operation *op) {
    TPU_CHECK_OP(op->getNumOperands() == 1, "expect 1 operand");
    TPU_CHECK_OP(op->getNumResults() == 1, "expect 1 result");
    auto src_ty = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op->getResult(0).getType());
    auto some_layout = getLayout(op->getOperand(0));
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    if (dyn_cast<arith::TruncFOp>(op)) {
      TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == 32 &&
                       (dst_ty.getElementTypeBitWidth() == 16 ||
                        dst_ty.getElementTypeBitWidth() == 8),
                   "Only 32-bit to 8-bit or 16-bit truncation supported");
    } else {
      TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == 32,
                   "Only 32-bit truncation supported");
    }
    auto &layout = *some_layout;
    bool select_native = allUsersRequireNativeTiling(op->getResult(0));
    auto src_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                   layout.implicit_dim());
    auto dst_layout = VectorLayout(
        dst_ty.getElementTypeBitWidth(), layout.offsets(),
        select_native ? nativeTiling(dst_ty.getElementTypeBitWidth())
                      : default_tiling_,
        layout.implicit_dim());
    setLayout(op, src_layout, dst_layout);
    return success();
  }

  LogicalResult inferElementwise(Operation *op, bool check_bitwidth = true) {
    TPU_CHECK_OP(op->getNumResults() == 1, "only one result supported");
    TPU_CHECK_OP(op->getNumOperands() > 0,
                 "elementwise ops with no operands unsupported");
    // Elementwise operators can be parameterized by both scalars and shaped
    // types, so make sure we infer layout based on a shaped-typed operand.
    std::optional<VectorLayout> out_layout_candidate;
    std::optional<VectorLayout> out_layout;
    SmallVector<std::optional<Layout>, 4> in_layouts;
    int64_t bit_width = -1;
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      if (auto vty = dyn_cast<VectorType>(op->getOperand(i).getType())) {
        if (bit_width == -1) {
          bit_width = vty.getElementTypeBitWidth();
        }
        TPU_CHECK_OP(
            !check_bitwidth || bit_width == vty.getElementTypeBitWidth(),
            "Generic elementwise rule only supports operands of same width");
        auto some_layout = getLayout(op->getOperand(i));
        TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
        auto &layout = *some_layout;
        // If the input is fully replicated, don't use it to commit to any
        // layout. Replicated values are easy to relayout.
        if (is_fully_replicated(some_layout)) {
          in_layouts.push_back(std::nullopt);
          out_layout_candidate = layout;
          continue;
        }
        if (!out_layout) {
          // TODO(apaszke): There are probably smarter ways to choose layout.
          out_layout = layout;
          in_layouts.push_back(some_layout);
        } else {
          if (auto new_out =
                  VectorLayout::join(layout, *out_layout, vty.getShape())) {
            out_layout = *new_out;
            in_layouts.push_back(some_layout);
          } else {
            // When we detect a layout conflict we cannot reconcile, we remove
            // any replication bits that might have been present in out_layout,
            // since there is no guarantee that the conflicting inputs could
            // even become replicated.
            out_layout =
                VectorLayout(out_layout->bitwidth(),
                             {out_layout->offsets()[0].value_or(0),
                              out_layout->offsets()[1].value_or(0)},
                             out_layout->tiling(), out_layout->implicit_dim());
            in_layouts.push_back(std::nullopt);
          }
        }
      } else {
        TPU_CHECK_OP(op->getOperand(i).getType().isSignlessIntOrIndexOrFloat(),
                     "expected only vector and scalar operands");
        in_layouts.push_back({kNoLayout});
      }
    }
    Layout final_out_layout = std::nullopt;
    if (auto out_vty = dyn_cast<VectorType>(op->getResult(0).getType())) {
      TPU_CHECK_OP(
          !check_bitwidth || bit_width == out_vty.getElementTypeBitWidth(),
          "Generic elementwise rule can't change element type width");
      if (out_layout) {
        final_out_layout = *out_layout;
      } else if (out_layout_candidate) {
        final_out_layout = *out_layout_candidate;
      } else {
        op->emitOpError(
            "Elementwise op has no vector operands but returns a vector?");
        return failure();
      }
    }
    CHECK_EQ(in_layouts.size(), op->getNumOperands()) << Print(op);
    SmallVector<Layout, 4> final_in_layouts;
    for (int i = 0; i < in_layouts.size(); ++i) {
      if (in_layouts[i]) {
        final_in_layouts.push_back(*in_layouts[i]);
      } else {
        final_in_layouts.push_back(final_out_layout);
      }
    }
    setLayout(op, final_in_layouts, final_out_layout);
    return success();
  }

  LogicalResult inferMatmul(Operation *op) {
    auto get_operand_layout =
        [&](Value v, llvm::StringRef operand_name,
            std::optional<int64_t> major_multiple = std::nullopt,
            std::optional<int64_t> minor_multiple =
                std::nullopt) -> std::optional<VectorLayout> {
      auto layout = getLayout(v);
      if (!layout.has_value()) {
        op->emitOpError("Internal error: assert failed: Operand ")
            << operand_name << " has no vector layout";
        return std::nullopt;
      }
      auto vty = cast<VectorType>(v.getType());
      auto tiling = nativeTiling(vty.getElementTypeBitWidth());
      auto shape = vty.getShape().take_back(2);
      if (shape[0] % major_multiple.value_or(tiling[0]) != 0 ||
          shape[1] % minor_multiple.value_or(tiling[1]) != 0) {
        op->emitOpError("Matmul operand")
            << operand_name << " must have a shape divisible by ("
            << major_multiple.value_or(tiling[0]) << ", "
            << minor_multiple.value_or(tiling[1]) << "), but got: (" << shape[0]
            << ", " << shape[1] << ")";
        return std::nullopt;
      }
      // Override tiling to match the native one.
      return VectorLayout(layout->bitwidth(), {0, 0}, tiling,
                          ImplicitDim::kNone);
    };
    auto res_ty = dyn_cast<VectorType>(op->getResult(0).getType());
    TPU_CHECK_OP(res_ty, "only vector results supported");
    TPU_CHECK_OP(res_ty.getElementTypeBitWidth() == kNativeBitwidth,
                 "only 32-bit matmul results supported");
    std::array<Layout, 3> in_layout;
    CHECK_EQ(op->getNumOperands(), 3);
    std::optional<int64_t> lhs_major_multiple;
    std::optional<int64_t> rhs_major_multiple;
    // We don't restrict the first lhs axis when the data is not packed.
    if (cast<VectorType>(op->getOperand(0).getType())
            .getElementTypeBitWidth() == kNativeBitwidth) {
      lhs_major_multiple = 1;
    }
    // We don't restrict the first rhs axis when the data is not packed.
    if (cast<VectorType>(op->getOperand(1).getType())
            .getElementTypeBitWidth() == kNativeBitwidth) {
      rhs_major_multiple = 1;
    }
    in_layout[0] =
        get_operand_layout(op->getOperand(0), "lhs", lhs_major_multiple, 1);
    if (!in_layout[0].has_value()) {
      return failure();
    }
    in_layout[1] =
        get_operand_layout(op->getOperand(1), "rhs", rhs_major_multiple, 1);
    if (!in_layout[1].has_value()) {
      return failure();
    }
    in_layout[2] = get_operand_layout(op->getOperand(2), "result", 1, 1);
    if (!in_layout[2].has_value()) {
      return failure();
    }
    setLayout(op, in_layout,
              VectorLayout(kNativeBitwidth, {0, 0}, default_tiling_,
                           ImplicitDim::kNone));
    return success();
  }
  LogicalResult infer(tpu::PRNGRandomBitsOp op) {
    auto res_ty = dyn_cast<VectorType>(op->getResult(0).getType());
    TPU_CHECK_OP(res_ty.getElementTypeBitWidth() == kNativeBitwidth,
                 "only 32-bit random bit generation supported");
    // TODO: b/342054464 - Support implicit dims for PRNGRandomBitsOp.
    LayoutOffsets offsets = {0, 0};
    setOutLayout(op, VectorLayout(
        kNativeBitwidth, offsets, nativeTiling(kNativeBitwidth),
                                  ImplicitDim::kNone));
    return success();
  }

  bool allUsersRequireNativeTiling(Value x) {
    for (OpOperand &operand : x.getUses()) {
      if (isa<vector::ContractionOp, tpu::MatmulOp>(operand.getOwner())) {
        continue;
      }
      if (auto transpose = dyn_cast<vector::TransposeOp>(operand.getOwner())) {
        auto perm = transpose.getPermutation();
        auto rank = perm.size();
        // Only permutations that actually swap the last two dims need it.
        if (rank >= 2 && perm[rank - 1] == rank - 2 &&
            perm[rank - 2] == rank - 1) {
          continue;
        }
        // Fall through.
      }
      return false;
    }
    return true;
  }

  LogicalResult assumeLayoutsForBlockArgs(Block &block,
                                          ArrayRef<Layout> layouts) {
    auto op = block.getParentOp();
    if (layouts.size() != block.getNumArguments()) {
      return op->emitOpError(
          "Block arguments must have the same number of layouts");
    }
    // Use tpu.assume_layout to annotate every block argument with the layout of
    // the corresponding operand and replace all uses of the block argument with
    // the result of tpu.assume_layout.
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(op->getLoc(), &block);
    for (auto [iter_arg, layout] :
         llvm::zip_equal(block.getArguments(), layouts)) {
      if (!dyn_cast<VectorType>(iter_arg.getType())) {
        continue;
      }
      if (llvm::any_of(iter_arg.getUsers(), [](Operation *user) {
            return isa<tpu::AssumeLayoutOp>(user);
          })) {
        return op->emitOpError("Expected no assume layout for block arguments");
      }
      auto assume_layout_op =
          builder.create<AssumeLayoutOp>(iter_arg.getType(), iter_arg);
      setLayout(assume_layout_op, layout, layout);
      iter_arg.replaceUsesWithIf(assume_layout_op, [&](OpOperand &operand) {
        return operand.getOwner() != assume_layout_op;
      });
    }
    return success();
  }

  void clearBlockLayouts(Block &block) {
    block.walk([&](Operation *op) {
      // We need to remove assume_layout ops in each block. Otherwise, we will
      // create extra assume_layout ops for nested blocks.
      if (auto assume_op = dyn_cast<tpu::AssumeLayoutOp>(op)) {
        assume_op.getResult().replaceAllUsesWith(assume_op.getInput());
        assume_op->erase();
        return WalkResult::advance();
      }
      op->removeAttr("in_layout");
      op->removeAttr("out_layout");
      return WalkResult::advance();
    });
  }

  void setInLayout(Operation *op, ArrayRef<Layout> in) {
    CHECK_EQ(in.size(), op->getNumOperands()) << Print(op);
    SmallVector<Attribute, 4> in_attrs;
    in_attrs.reserve(in.size());
    for (const Layout &p : in) {
      in_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
    }
    op->setAttr("in_layout", ArrayAttr::get(op->getContext(), in_attrs));
  }

  void setOutLayout(Operation *op, Layout out) {
    setOutLayout(op, ArrayRef<Layout>(out));
  }

  void setOutLayout(Operation *op, ArrayRef<Layout> out) {
    SmallVector<Attribute, 4> out_attrs;
    out_attrs.reserve(out.size());
    for (const Layout &p : out) {
      out_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
    }
    op->setAttr("out_layout", ArrayAttr::get(op->getContext(), out_attrs));
  }

  void setLayout(Operation *op, Layout in, Layout out) {
    setLayout(op, ArrayRef<Layout>(in), ArrayRef<Layout>(out));
  }

  void setLayout(Operation *op, ArrayRef<Layout> in, Layout out) {
    setLayout(op, in, ArrayRef<Layout>(out));
  }

  void setLayout(Operation *op, Layout in, ArrayRef<Layout> out) {
    setLayout(op, ArrayRef<Layout>(in), out);
  }

  void setLayout(Operation *op, ArrayRef<Layout> in, ArrayRef<Layout> out) {
    setInLayout(op, in);
    setOutLayout(op, out);
  }

  SmallVector<Layout, 4> getInLayout(Operation *op) {
    CHECK(op);
    CHECK(op->getAttr("in_layout"));
    auto in_attrs = op->getAttrOfType<ArrayAttr>("in_layout").getValue();
    CHECK_EQ(in_attrs.size(), op->getNumOperands());
    SmallVector<Layout, 4> in_layouts;
    in_layouts.reserve(op->getNumOperands());
    for (int i = 0; i < op->getNumOperands(); ++i) {
      in_layouts.push_back(cast<VectorLayoutAttr>(in_attrs[i]).getLayout());
    }
    return in_layouts;
  }

  SmallVector<Layout, 4> getOutLayout(Operation *op) {
    CHECK(op);
    CHECK(op->getAttr("out_layout"));
    auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
    CHECK_EQ(out_attrs.size(), op->getNumResults());
    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    for (int i = 0; i < op->getNumResults(); ++i) {
      out_layouts.push_back(cast<VectorLayoutAttr>(out_attrs[i]).getLayout());
    }
    return out_layouts;
  }

  Layout getLayout(Value v) {
    auto op = v.getDefiningOp();
    CHECK(op);
    auto op_result = dyn_cast<OpResult>(v);
    CHECK(op_result);
    auto result_index = op_result.getResultNumber();
    auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
    CHECK(out_attrs.size() > result_index);
    return cast<VectorLayoutAttr>(out_attrs[result_index]).getLayout();
  }

  SmallVector<Layout, 4> getLayoutFromOperands(Operation *op) {
    SmallVector<Layout, 4> layouts;
    layouts.reserve(op->getNumOperands());
    for (const auto &operand : op->getOperands()) {
      if (isa<VectorType>(operand.getType())) {
        layouts.push_back(getLayout(operand));
      } else {
        layouts.push_back(kNoLayout);
      }
    }
    return layouts;
  }

 private:
  std::optional<absl::Span<const int64_t>> verifyMemoryTiling(
      Operation *op, ArrayRef<xla::Tile> mem_tiling, int64_t rank,
      int8_t bitwidth) {
    const int packing = kNativeBitwidth / bitwidth;
    if (bitwidth == 32) {
      if (mem_tiling.size() != 1) {
        op->emitOpError("Only one-level tiling supported for 32-bit loads");
        return std::nullopt;
      }
    } else if (bitwidth < 32) {
      int64_t rows_per_tile;
      if (rank == 1) {
        if (mem_tiling.size() != 3) {
          op->emitOpError(
              "Only three-level tiling supported for 1D memory ops narrower "
              "than 32-bit");
          return std::nullopt;
        }
        auto first = mem_tiling[0].dimensions();
        auto second = mem_tiling[1].dimensions();
        if (first.size() != 1 || first[0] % (packing * target_shape_[1]) != 0) {
          op->emitOpError("Invalid first-level tile in 1D memory op");
          return std::nullopt;
        }
        rows_per_tile = first[0] / target_shape_[1];
        if (second.size() != 1 || second[0] != target_shape_[1]) {
          op->emitOpError("Invalid second-level tile in 1D memory op");
          return std::nullopt;
        }
      } else {
        if (mem_tiling.size() != 2) {
          op->emitOpError(
              "Only two-level tiling supported for 2D+ memory ops narrower "
              "than 32-bit");
          return std::nullopt;
        }
        auto first = mem_tiling[0].dimensions();
        rows_per_tile = first[0];
      }
      auto row_compressed = mem_tiling[mem_tiling.size() - 1].dimensions();
      if (row_compressed.size() != 2) {
        op->emitOpError("Expected 2D tiling for packed layout");
        return std::nullopt;
      }
      if (row_compressed[0] != (32 / bitwidth) || row_compressed[1] != 1) {
        op->emitOpError("Expected compressed packed layout");
        return std::nullopt;
      }
      if (row_compressed[0] > rows_per_tile) {
        op->emitOpError("Packing cannot introduce padding");
        return std::nullopt;
      }
    } else {
      op->emitOpError("Loads of types wider than 32-bit unsupported");
      return std::nullopt;
    }
    return mem_tiling[0].dimensions();
  }

  std::array<int64_t, 2> nativeTiling(int8_t bitwidth) {
    return {default_tiling_[0] * kNativeBitwidth / bitwidth,
            default_tiling_[1]};
  }

  std::array<int64_t, 2> target_shape_;
  std::array<int64_t, 2> default_tiling_;

  // Address alignment requirement, counted in 32-bit increments.
  static constexpr int64_t kVmemAlignment32 = 128;
  // TODO(apaszke): This is not really native on newer generations of TPUs.
  // Get rid of this temporary stopgap.
  static constexpr int8_t kNativeBitwidth = 32;
};

struct InferVectorLayoutPass
    : public impl::InferVectorLayoutPassBase<InferVectorLayoutPass> {
  InferVectorLayoutPass(int lane_count, int sublane_count) {
    this->sublane_count = sublane_count;
    this->lane_count = lane_count;
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    VectorLayoutInferer run({sublane_count, lane_count});
    if (run.infer(func).failed()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInferVectorLayoutPass(
    int lane_count, int sublane_count) {
  return std::make_unique<InferVectorLayoutPass>(lane_count, sublane_count);
}

}  // namespace mlir::tpu
