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

"""Tests for target library."""

from absl.testing import absltest
import numpy as np
from scipy.spatial import transform
from environments import target

# 90 degrees around x-axis, at (1.0, 2.0, 3.0)
_TCP_POSE = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0, 2.0],
    [0.0, 1.0, 0.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
])


class TargetTest(absltest.TestCase):

  def _get_test_config(self) -> target.TargetScoringConfig:
    return target.TargetScoringConfig(
        max_shaped_score=0.5,
        distance_score_weight=0.4,
    )

  def test_distance_score(self):
    config = self._get_test_config()

    # Outside threshold.
    target_info = target.TargetInfo(position=np.array([3.0, 2.0, 3.0]))
    self.assertAlmostEqual(
        target.distance_score(_TCP_POSE, target_info, config), 0.1
    )

    # Inside threshold.
    target_info = target.TargetInfo(position=np.array([1.0, 2.02, 3.0]))
    self.assertAlmostEqual(
        target.distance_score(_TCP_POSE, target_info, config), 1.0
    )

  def test_orientation_score(self):
    config = self._get_test_config()
    # Outside threshold.
    target_rotation = transform.Rotation.from_euler(
        'x', 60, degrees=True
    ).as_matrix()
    target_info = target.TargetInfo(
        position=np.array(
            [1.0, 2.0, 3.0],
        ),
        rotation=target_rotation,
    )
    self.assertAlmostEqual(
        target.orientation_score(_TCP_POSE, target_info, config), 0.454545454545
    )
    # Inside threshold.
    target_rotation = transform.Rotation.from_euler(
        'x', 80, degrees=True
    ).as_matrix()
    target_info = target.TargetInfo(
        position=np.array(
            [3.0, 2.0, 3.0],
        ),
        rotation=target_rotation,
    )
    self.assertAlmostEqual(
        target.orientation_score(_TCP_POSE, target_info, config), 1.0
    )
    # No rotation.
    target_info = target.TargetInfo(
        position=np.array(
            [3.0, 2.0, 3.0],
        ),
        rotation=None,
    )
    self.assertIsNone(target.orientation_score(_TCP_POSE, target_info, config))

  def test_mixed_score(self):
    config = self._get_test_config()
    self.assertAlmostEqual(target.mixed_score(0.3, 0.4, config), 0.192)
    self.assertAlmostEqual(target.mixed_score(0.3, None, config), 0.3)

  def test_in_solution_threshold(self):
    config = self._get_test_config()
    # No rotation.
    target_info = target.TargetInfo(
        position=np.array(
            [3.0, 2.0, 3.0],
        ),
        rotation=None,
    )
    self.assertFalse(
        target.in_solution_threshold(_TCP_POSE, target_info, config)
    )
    target_info = target.TargetInfo(
        position=np.array(
            [1.002, 2.0, 3.0],
        ),
        rotation=None,
    )
    self.assertTrue(
        target.in_solution_threshold(_TCP_POSE, target_info, config)
    )
    # In position threshold but not rotation.
    target_rotation = transform.Rotation.from_euler(
        'x', 30, degrees=True
    ).as_matrix()
    target_info = target.TargetInfo(
        position=np.array(
            [1.002, 2.0, 3.0],
        ),
        rotation=target_rotation,
    )
    self.assertFalse(
        target.in_solution_threshold(_TCP_POSE, target_info, config)
    )
    # In rotation threshold but not position.
    target_rotation = transform.Rotation.from_euler(
        'x', 80, degrees=True
    ).as_matrix()
    target_info = target.TargetInfo(
        position=np.array(
            [2.0, 2.0, 3.0],
        ),
        rotation=target_rotation,
    )
    self.assertFalse(
        target.in_solution_threshold(_TCP_POSE, target_info, config)
    )
    # In both thresholds.
    target_rotation = transform.Rotation.from_euler(
        'x', 80, degrees=True
    ).as_matrix()
    target_info = target.TargetInfo(
        position=np.array(
            [1.0, 2.002, 3.0],
        ),
        rotation=target_rotation,
    )
    self.assertTrue(
        target.in_solution_threshold(_TCP_POSE, target_info, config)
    )


if __name__ == '__main__':
  absltest.main()
