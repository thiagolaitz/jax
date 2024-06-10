# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests whether the frontend attributes added by the context manager are

correctly propagated to the jaxpr and mlir.
"""

from absl.testing import absltest
import jax
from jax._src import core
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.lax import lax
import jax.numpy as jnp


core.JaxprPpSettings.print_attributes = True


class FrontendAttributesTest(absltest.TestCase):

  def test_no_attributes(self):
    @jax.jit
    def f(a, b):
      return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    print("no attributes: ", f_lowered_text)
    self.assertNotIn("mhlo.frontend_attributes = {}", f_lowered_text)

  def test_f_jitted_jaxpr(self):
    @jax.jit
    def f(a, b):
      with jax.attributes(a="b"):
        return a + b

    f_jaxpr = jax.make_jaxpr(f)(1, 2)
    eqns = f_jaxpr.eqns
    self.assertIn("# [{'a': 'b'}]", str(eqns[0]))

  def test_f_jitted_mlir(self):
    @jax.jit
    def f(a, b):
      with jax.attributes(a="b"):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "b"}', f_lowered_text)

  def test_f_nonjitted_mlir(self):
    def f_add(a, b):
      return dispatch.apply_primitive(lax.add_p, a, b)

    arg1 = jax.numpy.arange(2)
    with jax.attributes(a="b"):
      self.assertIn(
          'mhlo.frontend_attributes = {a = "b"}',
          jax.jit(f_add).lower(arg1, arg1).as_text(),
      )

  def test_f_attributes_scope(self):
    with jax.attributes(a="b"):

      @jax.jit
      def f(a, b):
        return a + b

    # Expect no attributes
    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertNotIn("mhlo.frontend_attributes = {}", f_lowered_text)

  def test_f_attributes_overwrite(self):
    with jax.attributes(a="b"):

      @jax.jit
      def f(a, b):
        with jax.attributes(a="c"):
          return a + b

      f_lowered_text = f.lower(1.0, 2.0).as_text()
      self.assertIn('mhlo.frontend_attributes = {a = "c"}', f_lowered_text)

  def test_f_attributes_merge(self):
    with jax.attributes(key1="val1"):

      @jax.jit
      def f(a, b):
        with jax.attributes(key2="val2"):
          return a + b

      f_jaxpr = jax.make_jaxpr(f)(1.0, 2.0)
      eqns = f_jaxpr.eqns
      print("merge: ", str(eqns[0]))
      f_lowered_text = f.lower(1.0, 2.0).as_text()
      self.assertIn(
          'mhlo.frontend_attributes = {key1 = "val1", key2 = "val2"}',
          f_lowered_text,
      )

  def test_attr_caching_jit_mlir(self):
    @jax.jit
    def f_add_jit(a, b):
      return a + b

    with jax.attributes(b="c"):
      f_add_lowered1 = f_add_jit.lower(2.0, 3.0).as_text()
    # Expect no attributes in the mlir.
    f_add_lowered2 = f_add_jit.lower(1.0, 2.0).as_text()
    with jax.attributes(c="d"):
      f_add_lowered3 = f_add_jit.lower(4.0, 5.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {b = "c"}', f_add_lowered1)
    self.assertNotIn("mhlo.frontend_attributes = {}", f_add_lowered2)
    self.assertNotIn('mhlo.frontend_attributes = {b = "c"}', f_add_lowered2)
    self.assertNotIn('mhlo.frontend_attributes = {c = "d"}', f_add_lowered2)
    self.assertIn('mhlo.frontend_attributes = {c = "d"}', f_add_lowered3)

  def test_attr_caching_nonjit_mlir(self):
    def f_add(a, b):
      return dispatch.apply_primitive(lax.add_p, a, b)

    arg1 = jax.numpy.arange(2)
    arg2 = jax.numpy.arange(2) + 1
    arg3 = jax.numpy.arange(2) + 2
    with jax.attributes(b="c"):
      self.assertIn(
          'mhlo.frontend_attributes = {b = "c"}',
          jax.jit(f_add).lower(arg1, arg1).as_text(),
      )
    # Expect no attributes in the jaxpr.
    self.assertNotIn(
        "mhlo.frontend_attributes = {}",
        jax.jit(f_add).lower(arg2, arg2).as_text(),
    )
    self.assertNotIn(
        'mhlo.frontend_attributes = {b = "c"}',
        jax.jit(f_add).lower(arg2, arg2).as_text(),
    )
    self.assertNotIn(
        'mhlo.frontend_attributes = {c = "d"}',
        jax.jit(f_add).lower(arg2, arg2).as_text(),
    )

    with jax.attributes(c="d"):
      self.assertIn(
          'mhlo.frontend_attributes = {c = "d"}',
          jax.jit(f_add).lower(arg3, arg3).as_text(),
      )

  def test_axpy(self):
    @jax.jit
    def axpy(a, x, y):
      with jax.attributes(a="b"):
        return a * x + y

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}',
        axpy.lower(1.0, 2.0, 3.0).as_text(),
    )

  def test_while(self):
    @jax.jit
    def f(a):
      with jax.attributes(a="b"):
        return jax.lax.while_loop(lambda x: x < 10, lambda x: x + 1, a)

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(1.0).as_text()
    )

  def test_nested_jit(self):
    @jax.jit
    def f(x, y):
      with jax.attributes(a="b"):
        z = x * y

        @jax.jit
        def g(z):
          with jax.attributes(c="d"):
            return z**2 + 1

        return g(z)

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b", c = "d"}',
        f.lower(1.0, 2.0).as_text(),
    )

  def test_grad_jaxpr(self):
    @jax.jit
    def f(x, y):
      with jax.attributes(a="b"):
        return jax.grad(lambda x: x**3 + y**2 + jnp.sin(x))(x)

    f_jaxpr = jax.make_jaxpr(f)(1.0, 2.0)
    eqns = f_jaxpr.eqns
    for eq in eqns:
      self.assertIn("# [{'a': 'b'}]", str(eq))

  def test_grad_mlir(self):
    @jax.jit
    def f(x):
      with jax.attributes(a="b"):
        return jax.grad(lambda x: x**3 + x**2 + jnp.sin(x))(x)

    print("grad: ", f.lower(1.0).as_text())
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(1.0).as_text()
    )

  def test_pmap_jaxpr(self):
    @jax.jit
    def f(x):
      with jax.attributes(a="b"):
        return x / jax.lax.psum(x, "i")

    with jax.attributes(c="d"):
      f_pmap = jax.pmap(f, axis_name="i")
      f_jaxpr = jax.make_jaxpr(f_pmap)(jnp.arange(5))
      eqns = f_jaxpr.eqns
      for eq in eqns:
        self.assertIn("# [{'c': 'd', 'a': 'b'}]", str(eq))

  def test_pmap_mlir(self):
    @jax.jit
    def f(x):
      with jax.attributes(a="b"):
        return x / jax.lax.psum(x, "i")

    with jax.attributes(c="d"):
      f_pmap = jax.pmap(f, axis_name="i")
      print("pmap: ", f_pmap.lower(jnp.arange(5)).as_text())
      self.assertIn(
          'mhlo.frontend_attributes = {a = "b", c = "d"}',
          f_pmap.lower(jnp.arange(5)).as_text(),
      )

  def test_vmap_jaxpr(self):
    dct = {"a": 0.0, "b": jnp.arange(5.0)}

    @jax.jit
    def f(dct, x):
      with jax.attributes(a="b"):
        return dct["a"] + dct["b"] + x

    with jax.attributes(a="d"):
      f_vmap = jax.vmap(f, in_axes=({"a": None, "b": 0}, None))
      f_jaxpr = jax.make_jaxpr(f_vmap)(dct, 1.0)
      eqns = f_jaxpr.eqns
      for eq in eqns[1:]:
        print("eq: ", eq)
        self.assertIn("# [{'a': 'd'}]", str(eq))

  def test_vmap_mlir(self):
    @jax.jit
    def f(x, y):
      with jax.attributes(a="b"):
        return (x + y, y * 2.0)

    f_vmap_jaxpr = jax.make_jaxpr(jax.vmap(f, in_axes=(0, None)))
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}',
        f_vmap_jaxpr.lower(jnp.arange(5.0), 1.0).as_text(),
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
