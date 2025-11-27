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

"""Target scoring library."""

import dataclasses

from flax import struct
import numpy as np
from scipy.spatial import transform

from environments import data_pb2
from environments import util


class TargetInfo(struct.PyTreeNode):
  """Info for a position + orientation target."""

  position: np.ndarray  # (3,)
  rotation: np.ndarray | None = None  # (3,3)

  def __eq__(self, other):
    if isinstance(other, TargetInfo):
      return np.allclose(self.position, other.position) and np.allclose(
          self.rotation, other.rotation
      )
    return False

  def to_proto(self) -> data_pb2.TargetInfo:
    return data_pb2.TargetInfo(
        position=util.to_proto(self.position),
        rotation=util.to_proto(self.rotation),
    )

  @classmethod
  def from_proto(cls, proto: data_pb2.TargetInfo):
    return cls(
        position=util.from_proto(proto.position),
        rotation=util.from_proto(proto.rotation),
    )


@dataclasses.dataclass
class TargetState:
  """Target state.

  Attributes:
    done: Whether the target is done.
    robot: The robot that solved the target if the target is done.
  """

  done: bool = False
  robot: str | None = None


class TargetScoringConfig(struct.PyTreeNode):
  """Target scoring configuration.

  Attributes:
      distance_threshold: Distance at which a target is considered solved.
      angular_distance_threshold: Angular distance at which a target is
        considered solved.
      max_shaped_score: Maximum score when either distance or angular distance
        is outside the respective thresholds.
      distance_penalty_coefficient: Coefficient used in distance penalty
        calculations. See inline comments.
      distance_score_weight: Weight of distance score (1-distance_score_weight
        is the weight for orientation score).
  """

  distance_threshold: float = 0.025
  angular_distance_threshold: float = 15.0 / 180.0 * np.pi

  # Distance score (before scaling by max_shaped_score) =
  # distance_penalty_coefficient /
  # (distance_penalty_coefficient + distance)
  # where distance is the distance between robot tooltip and target centre.
  distance_penalty_coefficient: float = 0.5

  max_shaped_score: float = 0.0

  # Orientation score (before scaling by max_shaped_score) =
  # 1.0 -
  # max(((angular_distance - angular_distance_threshold) /
  # (pi - angular_distance_threshold)), 0.0)
  # where angular_distance is the angular distance between the tooltip rotation
  # and the target rotation (between [0, PI]).

  # Mixed score =
  # distance_score_weight * distance_score +
  # (1.0 - distance_score_weight) * orientation_score * distance_score
  distance_score_weight: float = 0.0


def _angular_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
  """Returns the angular distance between two poses."""
  rotation1 = transform.Rotation.from_matrix(pose1[:3, :3])
  rotation2 = transform.Rotation.from_matrix(pose2[:3, :3])
  diff = rotation1.inv() * rotation2
  return diff.magnitude()


def distance_score(
    tcp_pose: np.ndarray, target_info: TargetInfo, config: TargetScoringConfig
) -> float:
  tcp_position = tcp_pose[:3, 3]
  distance = np.linalg.norm(tcp_position - target_info.position)
  if distance < config.distance_threshold:
    return 1.0
  coeff = config.distance_penalty_coefficient
  return config.max_shaped_score * coeff / (coeff + distance)


def orientation_score(
    tcp_pose: np.ndarray, target_info: TargetInfo, config: TargetScoringConfig
) -> float | None:
  if target_info.rotation is None:
    return None
  tcp_rotation = tcp_pose[:3, :3]
  angular_distance = _angular_distance(tcp_rotation, target_info.rotation)
  if angular_distance < config.angular_distance_threshold:
    return 1.0
  threshold = config.angular_distance_threshold
  return config.max_shaped_score * (
      1.0 - max((angular_distance - threshold) / (np.pi - threshold), 0.0)
  )


def mixed_score(
    dist_score: float, ori_score: float | None, config: TargetScoringConfig
) -> float:
  if ori_score is None:
    return dist_score
  return (
      config.distance_score_weight * dist_score
      + (1.0 - config.distance_score_weight) * ori_score * dist_score
  )


def in_solution_threshold(
    tcp_pose: np.ndarray, target_info: TargetInfo, config: TargetScoringConfig
) -> bool:
  tcp_position = tcp_pose[:3, 3]
  distance = np.linalg.norm(tcp_position - target_info.position)
  if not distance < config.distance_threshold:
    return False
  if target_info.rotation is None:
    return True
  return (
      _angular_distance(tcp_pose[:3, :3], target_info.rotation)
      < config.angular_distance_threshold
  )
