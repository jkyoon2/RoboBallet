# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for util library."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np

from environments import util


class UtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('flat', np.array([0.1, 0.2, 0.3])),
      ('2d', np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])),
      ('singular_dim', np.array([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]])),
      ('int32s', np.array([1000000, -1000000], dtype=np.int32)),
      ('int8s', np.array([1, 2, 3, -1], dtype=np.int8)),
      ('float64s', np.array([0.1, 0.2, 0.3, -1.0], dtype=np.float64)),
  )
  def test_ndarray_reconstruction(self, arr):
    reconstructed = util.from_proto(util.to_proto(arr))
    np.testing.assert_array_almost_equal(arr, reconstructed)
    self.assertEqual(arr.dtype, reconstructed.dtype)

  @parameterized.named_parameters(
      ('empty', {}),
      ('single_element', {'data': np.array([0.1, 0.2, 0.3])}),
      (
          'flat',
          {
              'data': np.array([0.1, 0.2, 0.3]),
              'data2': np.array([0.1, 0.2, 0.3]),
          },
      ),
      (
          'all_nested',
          {
              'nested1': {
                  'data1': np.array([0.1, 0.2, 0.3]),
                  'data2': np.array([0.2, 0.3, 0.4]),
              },
              'nested2': {'data1': np.array([0.3, 0.4, 0.5])},
          },
      ),
      (
          'part_nested',
          {
              'nested1': {
                  'data1': np.array([0.1, 0.2, 0.3]),
                  'data2': np.array([0.2, 0.3, 0.4]),
              },
              'data2': np.array([0.3, 0.4, 0.5]),
          },
      ),
      (
          'mixed_types',
          {
              'nested1': {
                  'data1': np.array([0.1, 0.2, 0.3]),
                  'data2': np.array([1, 2, 3], dtype=np.int8),
              },
              'data2': np.array([0.3, 0.4, 0.5]),
          },
      ),
  )
  def test_pytree_reconstruction(self, tree):
    chex.assert_trees_all_close(
        tree, util.pytree_from_proto(util.pytree_to_proto(tree))
    )


if __name__ == '__main__':
  absltest.main()
