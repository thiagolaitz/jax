/* Copyright 2021 The JAX Authors.

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

#include "jaxlib/gpu/lu_pivot_kernels.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

template <typename T>
inline T CastNoOverflow(std::int64_t value,
                        const std::string& source = __FILE__) {
  if constexpr (sizeof(T) == sizeof(std::int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<T>::max()) [[unlikely]] {
      throw std::overflow_error{
          absl::StrFormat("%s: Value (=%d) exceeds the maximum representable "
                          "value of the desired type",
                          source, value)};
    }
    return static_cast<T>(value);
  }
}

ffi::Error LuPivotsToPermutationImpl(
    gpuStream_t stream, std::int32_t permutation_size,
    ffi::Buffer<ffi::DataType::S32> pivots,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> permutation) {
  auto dims = pivots.dimensions;
  std::int32_t pivot_size = CastNoOverflow<std::int32_t>(dims.back());
  std::int64_t batch_size = 1;
  if (dims.size() >= 2) {
    batch_size =
        absl::c_accumulate(dims.first(dims.size() - 1), 1, std::multiplies<>());
  }
  LaunchLuPivotsToPermutationKernel(stream, batch_size, pivot_size,
                                    permutation_size, pivots.data,
                                    permutation->data);
  if (auto status = JAX_AS_STATUS(gpuGetLastError()); !status.ok()) {
    return ffi::Error(static_cast<XLA_FFI_Error_Code>(status.code()),
                      std::string(status.message()));
  }
  return ffi::Error::Success();
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
