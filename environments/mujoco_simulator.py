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

"""RoboBallet simulator based on Mujoco."""

from collections.abc import Sequence
import copy
import dataclasses
import math
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import transform as sp_transform

from data import data_locator
from environments import ik_solver
from environments import simulator


# Number of integration steps per IK attempt.
_IK_MAX_ITERATIONS = 20
_IK_MAX_ATTEMPTS = 10


# IK tolerances. We are using very loose tolerances here because the solver we
# are using is quite slow, and we don't need accurate results anyways as this
# is just to filter out clearly bad target poses.
_IK_POS_TOL = 5e-2
_IK_ORIENTATION_TOL = 5e-2


# Mujoco uses groups to control rendering, so we designate a group for enabled
# markers, and another for disabled markers.
_SITE_GROUP_RENDERED = 2
_SITE_GROUP_UNUSED = 3

_GEOM_GROUP_VISUAL = 2
_GEOM_GROUP_COLLISION = 3


def _to_polar_angles(rel_pos: np.ndarray) -> tuple[float, float, float]:
  """Converts relative positions to distance, azimuth, elevation."""
  x, y, z = rel_pos[0], rel_pos[1], rel_pos[2]
  distance = np.linalg.norm(rel_pos)
  elevation = math.asin(z / distance) * 180 / math.pi
  azimuth = math.atan2(y, x) * 180 / math.pi
  return distance, azimuth, elevation


@dataclasses.dataclass(frozen=True)
class MujocoWorldConfig:
  """Configuration for MujocoWorld."""

  xml_path: str
  robot_names: list[str] = dataclasses.field(
      default_factory=lambda: [f'panda{i}' for i in range(1, 9)]
  )

  robot_joint_names: list[str] = dataclasses.field(
      default_factory=lambda: [f'joint{i}' for i in range(1, 8)]
  )

  # In MuJoCo usually home qpos are specified as a keyframe, but we can't
  # use that because keyframes are thrown away with the attach API. We specify
  # them here.
  robot_home_qpos: dict[str, list[float]] = dataclasses.field(
      default_factory=lambda: {
          'panda': [0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0]}
  )

  # Bodies in the world that we consider to be objects (object name to root body
  # name mappings).
  objects: dict[str, str] = dataclasses.field(
      default_factory=lambda: {
          'workpiece': 'workpiece'
      }
  )

  # Note that the often quoted max reach figure is 855mm, but that's on the
  # table surface, and in 3D it can actually reach further. Adding up all robot
  # segment lengths, we get 1.228m, and we add a bit more here for the end
  # effector. This is only used as an early exit for IK (if the target is more
  # than this far away, there's no point trying).
  robot_max_reach: float = 1.3


@dataclasses.dataclass(frozen=False)
class MujocoRobotInfo:
  """Mujoco robot information."""
  name: str  # Name of the robot and also the root body.
  lower_joint_limits: np.ndarray
  upper_joint_limits: np.ndarray
  max_reach: float


def _preprocess_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Preprocess the spec so that each geom has a name."""
  # We need every geom to have a name because the IK solver uses geom names to
  # define collision pairs.
  bodies = [spec.worldbody]
  geom_global_id = 0
  while bodies:
    body = bodies.pop()
    for geom in body.geoms:
      if not geom.name:
        geom.name = f'geom_{body.name}_{geom_global_id}'
        geom_global_id += 1
    bodies.extend(body.find_all('body'))
  return spec


class MujocoWorld(simulator.Simulator):
  """Mujoco simulator implementation."""

  def __init__(self, config: MujocoWorldConfig):
    self._config = config
    path = data_locator.get_data_path(config.xml_path)
    self._spec = _preprocess_spec(mujoco.MjSpec.from_file(path))
    self._model = self._spec.compile()
    self._data = mujoco.MjData(self._model)

    robot_infos = []
    for robot_name in self._config.robot_names:
      try:
        robot_body = self._model.body(robot_name)
      except KeyError:
        continue
      joints = self._get_robot_joints(robot_body.name)
      robot_info = MujocoRobotInfo(
          name=robot_body.name,
          lower_joint_limits=np.array(
              [
                  joint.range[0] for joint in joints
              ],
              dtype=np.float32),
          upper_joint_limits=np.array(
              [
                  joint.range[1] for joint in joints
              ],
              dtype=np.float32),
          max_reach=self._config.robot_max_reach,
      )
      # Initialize to home joint configurations.
      for prefix in self._config.robot_home_qpos:
        if robot_name.startswith(prefix):
          self._data.qpos[
              joints[0].qposadr[0]:(joints[-1].qposadr[0] + 1)] = np.array(
                  self._config.robot_home_qpos[prefix], dtype=np.float32)
        break
      else:
        raise ValueError(
            f'Robot {robot_name} does not start with any of '
            f'{self._config.robot_home_qpos.keys()}'
        )
      robot_infos.append(robot_info)
    robot_infos = sorted(robot_infos, key=lambda x: x.name)
    self._robots = {robot.name: robot for robot in robot_infos}

    # We also treat robot bases as objects.
    object_mappings = self._config.objects.copy()
    for robot_name, robot_info in self._robots.items():
      object_mappings[robot_name] = f'{robot_info.name}'

    self._objects = {}  # object name -> root body name.
    for object_name, body_name in object_mappings.items():
      self._objects[object_name] = self._model.body(body_name).name

    self._next_body_id = 0

    self._renderer = None

    # Whether we have updated qpos-derived quantities.
    self._kinematics_valid = False

    # MuJoCo requires a recompilation every time objects are added or removed.
    # We do delayed recompilation to speed things up if the caller wants to
    # add/remove many objects in one go.
    self._need_recompile = False

    # Name of all static collision geoms.
    self._static_geom_names = self._get_static_geom_names()

  def __copy__(self):
    ret = object.__new__(type(self))

    # First we copy things that don't need special treatment.
    for k, v in self.__dict__.items():
      if k in ('_data', '_spec', '_model', '_renderer'):
        continue
      setattr(ret, k, copy.deepcopy(v))

    # Then we handle the special cases.
    ret._spec = self._spec.copy()

    # Copying the model should work here in theory, but seems to corrupt the
    # model in a way that makes RemoveObject fail (when recompiling the modified
    # spec based on the copied model, it fails to preserve state). Compiling
    # the spec again seems to work and is recommended by mujoco people (though
    # it may be slower - benchmarking required).
    ret._model = ret._spec.compile()

    # Renderer is constructed on first use when necessary. We don't want to copy
    # it around because we don't need to render most worlds.
    ret._renderer = None

    # The recommended way to copy MjData is to copy the state and run forward.
    # So we just copy the joint configs and set kinematics to invalid.
    ret._data = mujoco.MjData(ret._model)
    ret._data.qpos[:] = self._data.qpos[:]
    ret._data.mocap_pos[:] = self._data.mocap_pos[:]
    ret._data.mocap_quat[:] = self._data.mocap_quat[:]
    ret._kinematics_valid = False
    ret._need_recompile = False
    return ret

  def _get_static_geom_names(self) -> list[str]:
    """Returns the names of all static collision geoms in the model."""
    self._update_model()
    all_geoms = set()
    for body_id in range(self._model.nbody):
      body = self._model.body(body_id)
      geoms_start = body.geomadr[0]
      geoms_end = geoms_start + body.geomnum[0]
      for geom_id in range(geoms_start, geoms_end):
        geom = self._model.geom(geom_id)
        if (geom.conaffinity[0] != 0 and geom.contype[0] != 0):
          assert geom.name, (f'Geom {geom_id} has no name (body='
                             f'{body.name}, id={body_id}).')
          all_geoms.add(geom.name)

    for robot_name in self._robots:
      for geom in self._get_all_collision_geoms_from_body(
          self._model.body(robot_name)):
        all_geoms.remove(geom.name)
    return list(all_geoms)

  def _update_model(self):
    if self._need_recompile:
      self._model, self._data = self._spec.recompile(self._model, self._data)
      self._need_recompile = False
      self._static_geom_names = self._get_static_geom_names()

  def _update_kinematics(self):
    """Updates the qpos-derived quantities if necessary."""
    self._update_model()

    if not self._kinematics_valid:
      mujoco.mj_kinematics(self._model, self._data)
      mujoco.mj_collision(self._model, self._data)
      self._data.qvel[:] = 0.0
      self._kinematics_valid = True

  def _to_transform(self, xpos: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    assert xpos.shape == (3,), f'Expected 3D position, got {xpos.shape}.'
    assert xmat.shape == (9,), (
        f'Expected 3x3 rotation matrix, got {xmat.shape}.')
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.reshape(xmat, (3, 3))
    mat[:3, 3] = xpos
    return mat

  def _from_transform(self, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert mat.shape == (4, 4), f'Expected 4x4 transform, got {mat.shape}.'
    return mat[:3, 3], mat[:3, :3]

  def _get_robot_joints(self, robot_name: str) -> list[Any]:
    joints = []
    for joint_name in self._config.robot_joint_names:
      full_joint_name = f'{robot_name}/{joint_name}'
      joints.append(self._model.jnt(full_joint_name))
    return joints

  def _get_robot_qpos_indices(self, robot_name: str) -> list[int]:
    return [int(joint.qposadr[0])
            for joint in self._get_robot_joints(robot_name)]

  def _get_site_transform(self, site) -> np.ndarray:
    self._update_kinematics()
    return self._to_transform(self._data.site_xpos[site.id],
                              self._data.site_xmat[site.id])

  def _get_body_transform(self, body) -> np.ndarray:
    self._update_kinematics()
    return self._to_transform(self._data.xpos[body.id],
                              self._data.xmat[body.id])

  def _get_all_children_of_body(self, root_body_id: int) -> Sequence[int]:
    """Returns all children of a body (recursive)."""
    self._update_model()
    # MuJoCo only stores parents of bodies, not children, so we build a reverse
    # mapping first.
    children_by_parent = {}
    for body_id in range(self._model.nbody):
      parent_id = self._model.body_parentid[body_id]
      # Exclude bodies that are their own parent.
      if body_id == parent_id:
        continue
      children_by_parent.setdefault(parent_id, []).append(body_id)
    bodies_to_expand = [root_body_id]
    children = []
    while bodies_to_expand:
      body_id = bodies_to_expand.pop()
      children.append(body_id)
      if body_id in children_by_parent:
        bodies_to_expand.extend(children_by_parent[body_id])
    return children

  def _robot_from_body_id(self, body_id: int) -> str | None:
    """Returns the name of the robot that contains the given body (if any)."""
    self._update_model()
    body = self._model.body(body_id)
    while body.id != 0:
      if body.name in self._robots:
        return body.name
      body = self._model.body(body.parentid[0])
    return None

  def _get_all_collision_geoms_from_body(self, body) -> Sequence[Any]:
    """Returns all collision geoms from a body (recursive)."""
    self._update_kinematics()
    geoms = []
    for body_id in self._get_all_children_of_body(body.id):
      geoms_start = self._model.body_geomadr[body_id]
      geoms_end = geoms_start + self._model.body_geomnum[body_id]
      for geom_id in range(geoms_start, geoms_end):
        # Only if they are collision geoms.
        if (self._model.geom_conaffinity[geom_id] != 0 and
            self._model.geom_contype[geom_id] != 0):
          geoms.append(self._model.geom(geom_id))
    return geoms

  def GetObjects(self) -> set[str]:
    """Returns the set of objects in the world."""
    return set(self._objects.keys())

  def GetWorldTransform(self, obj_b: str) -> np.ndarray:
    """Returns the transform from the world root to obj_b."""
    self._update_model()
    if obj_b not in self._objects:
      raise ValueError(f'Object {obj_b} not known.')
    return self._get_body_transform(self._model.body(self._objects[obj_b]))

  def SetWorldTransform(self, obj: str, transform: np.ndarray) -> None:
    """Sets the transform from the world root to obj."""
    self._update_model()
    if obj not in self._objects:
      raise ValueError(f'Object {obj} not known.')
    mocap_id = self._model.body(self._objects[obj]).mocapid
    if mocap_id == -1:
      raise ValueError(f'Object {obj} does not have a mocap tag. We can only '
                       'move objects with mocap tags.')
    self._data.mocap_pos[mocap_id] = transform[:3, 3]

    rotation = sp_transform.Rotation.from_matrix(transform[:3, :3])
    rotation = rotation.as_quat(canonical=False)
    self._data.mocap_quat[mocap_id] = np.array(
        [rotation[3], rotation[0], rotation[1], rotation[2]],
        dtype=transform.dtype)
    self._kinematics_valid = False

  def AddBox(
      self,
      dims: np.ndarray,
      color: np.ndarray = simulator.WHITE_COLOR,
      prefix: str = 'box_',
  ) -> str:
    """Adds a box to the world."""

    half_dims = dims / 2.0

    # See https://upload.wikimedia.org/wikipedia/commons/2/2d/Mesh_fv.jpg
    vertices = np.zeros((8, 3), dtype=np.float32)
    vertices[(1, 2, 5, 6), 0] = half_dims[0]  # +x
    vertices[(0, 3, 4, 7), 0] = -half_dims[0]  # -x
    vertices[(4, 5, 6, 7), 1] = half_dims[1]  # +y
    vertices[(0, 1, 2, 3), 1] = -half_dims[1]  # -y
    vertices[(0, 1, 4, 5), 2] = half_dims[2]  # +z
    vertices[(2, 3, 6, 7), 2] = -half_dims[2]  # -z
    # This needs to be right-handed to match our MuJoCo models.
    faces = np.array([
        [0, 5, 4], [0, 1, 5],
        [1, 6, 5], [1, 2, 6],
        [2, 7, 6], [2, 3, 7],
        [3, 4, 7], [3, 0, 4],
        [4, 5, 6], [4, 6, 7],
        [0, 2, 1], [0, 3, 2]
    ], dtype=np.int32)
    return self.AddMesh(vertices, faces, color, prefix=prefix)

  def AddMesh(
      self,
      vertices: np.ndarray,
      faces: np.ndarray,
      color: np.ndarray = simulator.WHITE_COLOR,
      prefix: str = 'mesh_'
  ) -> str:
    """Adds a mesh to the world."""
    new_body_id = f'{prefix}{self._next_body_id}'
    self._next_body_id += 1
    new_mesh_name = f'mesh_for_body_{new_body_id}'

    mesh = self._spec.add_mesh()
    mesh.name = new_mesh_name
    mesh.uservert = vertices.flatten()
    mesh.userface = faces.flatten()

    new_body = self._spec.worldbody.add_body(name=new_body_id, mocap=True)

    # Mujoco wants RGBA colors.
    color = np.concatenate([color, [1.0]], axis=-1)
    new_body.add_geom(name=f'{new_body_id}/geom',
                      type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=new_mesh_name,
                      rgba=color)
    self._objects[new_body_id] = new_body_id
    self._kinematics_valid = False
    self._need_recompile = True
    return new_body_id

  def RemoveObject(self, obj_id: str) -> None:
    """Removes an object from the world."""
    self._update_model()
    if obj_id not in self._objects:
      raise ValueError(f'Object {obj_id} not known.')
    body = self._model.body(self._objects[obj_id])
    spec_body = self._spec.body(body.name)
    self._spec.delete(spec_body)
    del self._objects[obj_id]
    if obj_id in self._robots:
      del self._robots[obj_id]
    self._kinematics_valid = False
    self._need_recompile = True

  # Returns a list of meshes, each of which has a Nx3 array of vertices and a
  # Mx3 array of faces.
  def GetMeshes(self, obj_id: str) -> list[simulator.Mesh]:
    self._update_kinematics()
    if obj_id not in self._objects:
      raise ValueError(f'Object {obj_id} not known.')
    root_body = self._model.body(self._objects[obj_id])
    geoms = self._get_all_collision_geoms_from_body(root_body)
    meshes = []
    for geom in geoms:
      geom_xpos = self._data.geom_xpos[geom.id]
      geom_xmat = self._data.geom_xmat[geom.id]
      geom_pose = self._to_transform(geom_xpos, geom_xmat)
      if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = self._model.mesh(geom.dataid[0]).id
        vert_indices_begin = self._model.mesh_vertadr[mesh_id]
        vert_indices_end = (
            vert_indices_begin + self._model.mesh_vertnum[mesh_id])
        face_indices_begin = self._model.mesh_faceadr[mesh_id]
        face_indices_end = (
            face_indices_begin + self._model.mesh_facenum[mesh_id])
        vertices = self._model.mesh_vert[vert_indices_begin:vert_indices_end]
        faces = self._model.mesh_face[face_indices_begin:face_indices_end]
        vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))],
                                  axis=-1)
        vertices = vertices @ geom_pose.T
        mesh = simulator.Mesh(
            vertices=vertices[:, :3].astype(np.float32),
            faces=faces[:, [0, 2, 1]].astype(np.int32))
        meshes.append(mesh)
      else:
        raise NotImplementedError('Only meshes are supported.')
    return meshes

  def GetRobots(self) -> set[str]:
    return set(self._robots.keys())

  def GetRobotBaseTransform(self, robot: str) -> np.ndarray:
    self._update_model()
    return self._get_site_transform(self._model.site(f'{robot}/base_site'))

  def GetRobotTcpTransform(self, robot: str) -> np.ndarray:
    self._update_model()
    return self._get_site_transform(
        self._model.site(f'{robot}/tool/tool_tip_site'))

  def GetLowerJointPositionLimits(self, robot: str) -> np.ndarray:
    """Returns the lower joint position limits for the given robot."""
    return self._robots[robot].lower_joint_limits

  def GetUpperJointPositionLimits(self, robot: str) -> np.ndarray:
    """Returns the upper joint position limits for the given robot."""
    return self._robots[robot].upper_joint_limits

  def GetJointValues(self, robot: str) -> np.ndarray:
    """Returns the joint values for the given robot."""
    qpos_indices = self._get_robot_qpos_indices(robot)
    return self._data.qpos[qpos_indices[0]:(qpos_indices[-1] + 1)].copy()

  def SetJointValues(self, robot: str, values: np.ndarray) -> None:
    """Sets the joint values for the given robot."""
    assert (values <= self._robots[robot].upper_joint_limits).all()
    assert (values >= self._robots[robot].lower_joint_limits).all()
    qpos_indices = self._get_robot_qpos_indices(robot)
    self._data.qpos[qpos_indices[0]:(qpos_indices[-1] + 1)] = values
    self._kinematics_valid = False

  def ComputeIk(
      self, robot: str, world_t_target: np.ndarray, num_solutions: int,
      solver_configs: dict[str, float | bool] | None = None,
  ) -> list[np.ndarray]:
    """Computes the IK solutions for the given robot and target pose."""
    assert num_solutions == 1, 'Only one solution is supported.'
    self._update_kinematics()

    # Is it beyond our reach? Early exit.
    base_pose = self.GetRobotBaseTransform(robot)
    base_t_target = np.linalg.inv(base_pose) @ world_t_target
    if np.linalg.norm(base_t_target[:3, 3]) > self._robots[robot].max_reach:
      return []

    joint_ids = [joint.id for joint in self._get_robot_joints(robot)]
    start_joint_values = self.GetJointValues(robot)
    target_quat = sp_transform.Rotation.from_matrix(
        world_t_target[:3, :3]).as_quat(canonical=False)
    # Scipy uses scalar last, we need scalar first.
    target_quat = np.array([target_quat[3], target_quat[0],
                            target_quat[1], target_quat[2]],
                           dtype=world_t_target.dtype)
    robot_geom_names = [
        geom.name for geom in
        self._get_all_collision_geoms_from_body(self._model.body(robot))
    ]
    collision_pairs = [
        (
            robot_geom_names, self._static_geom_names
        ),
        (
            robot_geom_names, robot_geom_names
        ),
    ]
    solver = ik_solver.IkSolver(model=self._model,
                                controllable_joints_ids=joint_ids,
                                site_name=f'{robot}/tool/tool_tip_site',
                                collision_pairs=collision_pairs,
                                solver_configs=solver_configs)

    if solver_configs is None:
      solver_configs = {
          'max_attempts': _IK_MAX_ATTEMPTS,
          'max_iterations': _IK_MAX_ITERATIONS,
      }

    solution = None
    for attempt in range(solver_configs['max_attempts']):
      if attempt == 0:
        start_config = start_joint_values
      else:
        self.RandomlyMoveRobot(robot)
        start_config = self.GetJointValues(robot)

      maybe_joint_configs, _ = solver.solve(
          desired_global_pos_tcp=world_t_target[:3, 3],
          desired_global_quat_tcp=target_quat,
          initial_joint_configuration=start_config,
          linear_tol=_IK_POS_TOL,
          angular_tol=_IK_ORIENTATION_TOL,
          max_steps=solver_configs['max_iterations'],
          early_stop=True)
      if maybe_joint_configs is not None:
        self.SetJointValues(robot, maybe_joint_configs)
        in_collision = self.RobotsInCollision()
        if not in_collision:
          solution = maybe_joint_configs
          break
    self.SetJointValues(robot, start_joint_values)
    return [solution] if solution is not None else []

  def RandomlyMoveRobot(self, robot: str):
    """Randomly moves a robot to a collision-free pose."""
    self._update_kinematics()
    while True:
      self.SetJointValues(robot, np.random.uniform(
          low=self._robots[robot].lower_joint_limits,
          high=self._robots[robot].upper_joint_limits))
      if not self.RobotsInCollision():
        break

  def RobotsInCollision(self, margin: float = 0) -> set[str]:
    """Returns the set of robots in collision."""
    self._update_kinematics()
    assert margin == 0.0, 'Margin not implemented yet'
    in_collision = set()
    for contact_id in range(self._data.ncon):
      contact = self._data.contact[contact_id]
      if contact.dist <= margin:
        bodies = [
            self._model.body(self._model.geom(contact.geom[0]).bodyid[0]),
            self._model.body(self._model.geom(contact.geom[1]).bodyid[0])
        ]
        for body in bodies:
          maybe_robot = self._robot_from_body_id(body.id)
          if maybe_robot:
            in_collision.add(maybe_robot)
    return in_collision

  def RenderScene(self, markers: Sequence[simulator.Marker], width: int,
                  height: int, camera_pos: np.ndarray, lookat: np.ndarray,
                  fovy: float,
                  render_collision_geoms: bool = False) -> np.ndarray:
    self._update_kinematics()

    # In Mujoco setting FOV requires recompiling the model, so ideally don't do
    # that too often.
    need_recompile = False
    global_ = getattr(self._spec.visual, 'global')
    if fovy != global_.fovy:
      global_.fovy = fovy
      need_recompile = True

    # Do we have enough site markers we can use? If not add some sites that we
    # can then move around as markers. Having more marker sites than we need is
    # fine. We'll disable them later. If we add markers we need to recompile.
    if markers:
      highest_marker_site_name = f'frame_marker_{len(markers) - 1}'
      if not self._spec.site(highest_marker_site_name):
        for i in range(len(markers)):
          marker_name = f'frame_marker_{i}'
          if not self._spec.site(marker_name):
            self._spec.worldbody.add_site(name=marker_name,
                                          group=_SITE_GROUP_UNUSED)
        need_recompile = True

    if need_recompile:
      self._need_recompile = True

    self._update_kinematics()

    if not self._renderer or need_recompile:
      self._renderer = mujoco.Renderer(self._model, height, width)

    # We don't usually compute camera and lighting unless we are rendering.
    mujoco.mj_camlight(self._model, self._data)

    # Changing mjmodel after compilation is usually not recommended, but there's
    # no way to change site group (to enable/disable rendering them) without
    # recompiling, and recompiling is too slow. @tassa says it's fine to just
    # change the group of the sites in the model.
    for site_id in range(self._model.nsite):
      site = self._model.site(site_id)
      if site.name.startswith('frame_marker_'):
        marker_id = int(site.name.split('_')[-1])
        in_use = marker_id < len(markers)
        self._model.site(site_id).group = (
            _SITE_GROUP_RENDERED if in_use else _SITE_GROUP_UNUSED)
        if in_use:
          marker = markers[marker_id]
          assert isinstance(marker, simulator.FrameMarker)
          translation, rotation = self._from_transform(marker.pose)
          self._data.site_xpos[site_id] = translation
          self._data.site_xmat[site_id] = rotation.reshape((9,))

    camera = mujoco.MjvCamera()
    camera.lookat = lookat
    camera.distance, camera.azimuth, camera.elevation = _to_polar_angles(
        lookat - camera_pos)

    scene_option = mujoco.MjvOption()
    scene_option.frame |= mujoco.mjtFrame.mjFRAME_SITE
    scene_option.sitegroup[_SITE_GROUP_UNUSED] = False
    scene_option.sitegroup[_SITE_GROUP_RENDERED] = True

    if render_collision_geoms:
      scene_option.geomgroup[_GEOM_GROUP_VISUAL] = False
      scene_option.geomgroup[_GEOM_GROUP_COLLISION] = True

    self._renderer.update_scene(self._data, camera=camera,
                                scene_option=scene_option)
    return self._renderer.render()
