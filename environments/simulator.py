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

"""RobotBallet world simulator interface."""

import abc
from collections.abc import Sequence
import dataclasses

import numpy as np

WHITE_COLOR = np.array([1, 1, 1], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class Mesh:
  vertices: np.ndarray  # Nx3 array of vertices.
  faces: np.ndarray  # Mx3 array of faces (indices into vertices).


@dataclasses.dataclass(frozen=True)
class Marker:
  """Marker for rendering with the scene."""
  enabled: bool  # Whether the marker is enabled (rendered more prominently).


@dataclasses.dataclass(frozen=True)
class FrameMarker(Marker):
  """Marker for a coordinate frame."""
  pose: np.ndarray  # 4x4 transform from the world root to the frame.


class Simulator(abc.ABC):
  """Interface for a simulated world."""

  @abc.abstractmethod
  def GetObjects(self) -> set[str]:
    """Returns the set of objects in the world."""

  @abc.abstractmethod
  def GetWorldTransform(self, obj: str) -> np.ndarray:
    """Returns the transform from the world root to obj."""

  @abc.abstractmethod
  def SetWorldTransform(self, obj: str, transform: np.ndarray) -> None:
    """Sets the transform from the world root to obj."""

  @abc.abstractmethod
  def AddBox(self, dims: np.ndarray, color: np.ndarray = WHITE_COLOR) -> str:
    """Adds a box to the world."""

  @abc.abstractmethod
  def AddMesh(
      self,
      vertices: np.ndarray,
      faces: np.ndarray,
      color: np.ndarray = WHITE_COLOR,
  ) -> str:
    """Adds a mesh to the world."""

  @abc.abstractmethod
  def RemoveObject(self, obj_id: str) -> None:
    """Removes an object from the world."""

  @abc.abstractmethod
  def GetMeshes(self, obj_id: str) -> list[Mesh]:
    """Returns the meshes for the given object."""

  def GetMeshFaces(self, obj_id: str) -> np.ndarray:
    """Returns the mesh faces for the given object."""
    meshes = self.GetMeshes(obj_id)
    mesh_faces_vertices = []
    for mesh in meshes:
      for vertex_idx in mesh.faces.flatten():
        mesh_faces_vertices.append(mesh.vertices[vertex_idx])
    return np.array(np.stack(mesh_faces_vertices)).reshape((-1, 3, 3))

  @abc.abstractmethod
  def GetRobots(self) -> set[str]:
    """Returns the set of robots in the world."""

  @abc.abstractmethod
  def GetRobotBaseTransform(self, robot: str) -> np.ndarray:
    """Returns the transform from the world root to the robot base."""

  @abc.abstractmethod
  def GetRobotTcpTransform(self, robot: str) -> np.ndarray:
    """Returns the transform from the world root to the robot TCP."""

  @abc.abstractmethod
  def GetLowerJointPositionLimits(self, robot: str) -> np.ndarray:
    """Returns the lower joint position limits for the given robot."""

  @abc.abstractmethod
  def GetUpperJointPositionLimits(self, robot: str) -> np.ndarray:
    """Returns the upper joint position limits for the given robot."""

  @abc.abstractmethod
  def GetJointValues(self, robot: str) -> np.ndarray:
    """Returns the joint values for the given robot."""

  @abc.abstractmethod
  def SetJointValues(self, robot: str, values: np.ndarray) -> None:
    """Sets the joint values for the given robot."""

  @abc.abstractmethod
  def ComputeIk(
      self, robot: str, world_t_target: np.ndarray, num_solutions: int
  ) -> list[np.ndarray]:
    """Computes the IK solutions for the given robot and target pose."""

  @abc.abstractmethod
  def RobotsInCollision(self, margin: float = 0) -> set[str]:
    """Returns the set of robots in collision."""

  def RenderScene(self, markers: Sequence[Marker], width: int, height: int,
                  camera_pos: np.ndarray, lookat: np.ndarray, fovy: float
                  ) -> np.ndarray:
    """Renders the scene."""
    raise NotImplementedError()
