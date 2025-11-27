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

"""Tests for stats library."""

from absl.testing import absltest
from train import stats


class StatsTest(absltest.TestCase):

  def test_labels(self):
    trajectory_stats = [
        stats.TrajectoryStats(final_score=0.5,
                              num_targets_done=3,
                              episode_length=10,
                              episode_wall_time_s=2.3),
        stats.TrajectoryStats(final_score=0.8,
                              num_targets_done=3,
                              episode_length=15,
                              episode_wall_time_s=2.0),
        stats.TrajectoryStats(final_score=0.5,
                              num_targets_done=4,
                              episode_length=16,
                              episode_wall_time_s=1.9),
    ]
    aggregated = stats.mean_aggregate_stats(trajectory_stats)
    self.assertAlmostEqual(aggregated['final_score'], 0.6)
    self.assertAlmostEqual(aggregated['num_targets_done'], 3.33333, delta=1e-3)


if __name__ == '__main__':
  absltest.main()
