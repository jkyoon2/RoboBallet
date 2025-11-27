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

"""Library for statistics."""

from collections.abc import Sequence
import dataclasses
import statistics


@dataclasses.dataclass(frozen=False)
class TrainingStats:
  """Training statistics.

  Attributes:
    policy_loss: Actor deterministic policy gradient loss.
    twin_critic_loss: Twin critic loss.
    train_step_duration_s: Time for a core training step in seconds.
    sampling_wall_time_ms: Time for sampling a batch.
    total_step_time_ms: Total time for a step (sampling + batching + training).
    trajectory_network_age: How old the networks used to generate the samples
      are when the trajectories are sampled for training.
  """
  policy_loss: float = 0.0
  twin_critic_loss: float = 0.0
  sampling_wall_time_ms: float = 0.0
  total_step_time_ms: float = 0.0
  trajectory_network_age: int = 0


@dataclasses.dataclass(frozen=False)
class TrajectoryStats:
  """Stats associated with a trajectory.

  Attributes:
    final_score: Final score of the episode.
    total_reward: Total reward of the episode.
    total_collision_penalty: Total collision penalty of the episode.
    total_collision_robot_steps: Sum of number of robots in collision at each
      step of the episode.
    total_mean_acceleration: Total acceleration (mean of all DoF).
    num_targets_done: Number of targets done at the end of the episode.
    episode_length: Episode length in num steps.
    env_sampling_time_s: Environment sampling time.
    episode_wall_time_s: Episode wall time duration in seconds.
    episode_write_wall_time_s: Episode write time (to reverb).
    is_hindsight_replay: Whether this episode is "imagined" by hindsight. On a
      hindsight experience, all other stats are that of the actual trajectory.
    network_step: Network step of the policy used to generate the episode.
  """
  final_score: float = 0.0
  total_reward: float = 0.0
  total_collision_penalty: float = 0.0
  total_collision_robot_steps: int = 0
  total_mean_acceleration: float = 0.0
  num_targets_done: int = 0
  episode_length: int = 0
  env_sampling_time_s: float = 0.0
  episode_wall_time_s: float = 0.0
  episode_write_wall_time_s: float = 0.0
  is_hindsight_replay: int = 0
  network_step: int = 0


def mean_aggregate_stats(
    stats: Sequence[TrajectoryStats] | Sequence[TrainingStats]) -> dict[
        str, float]:
  """Returns the means of all values for each field."""
  assert stats
  list_of_values_by_field_name = {
      field.name: [] for field in dataclasses.fields(stats[0])
  }
  for stat in stats:
    for name, value in dataclasses.asdict(stat).items():
      list_of_values_by_field_name[name].append(float(value))
  return {
      name: float(statistics.mean(values))
      for name, values in list_of_values_by_field_name.items()
  }
