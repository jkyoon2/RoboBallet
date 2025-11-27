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

import copy

from absl.testing import absltest
import numpy as np

from data import data_locator
from environments import mujoco_simulator


class BaseEnvTest(absltest.TestCase):

  def make_simulator(self):
    full_path = data_locator.get_data_path('mujoco_world/4_pandas_world.xml')
    config = mujoco_simulator.MujocoWorldConfig(
        xml_path=full_path,
    )
    return mujoco_simulator.MujocoWorld(config)

  def test_loading(self):
    sim = self.make_simulator()
    self.assertCountEqual(sim.GetRobots(),
                          ['panda1', 'panda2', 'panda3', 'panda4'])

  def test_robot_manipulations(self):
    sim = self.make_simulator()
    lower = sim.GetLowerJointPositionLimits('panda1')
    upper = sim.GetUpperJointPositionLimits('panda1')
    current = sim.GetJointValues('panda1')
    self.assertEqual(lower.shape, (7,))
    self.assertEqual(upper.shape, (7,))
    self.assertEqual(current.shape, (7,))
    assert (current >= lower).all()
    assert (current <= upper).all()
    assert (lower < upper).all()
    np.testing.assert_allclose(
        current, np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    )
    with self.assertRaises(AssertionError):
      invalid_values = current.copy()
      invalid_values[0] = -10.0
      sim.SetJointValues('panda1', invalid_values)
    current = sim.GetJointValues('panda1')
    np.testing.assert_allclose(
        current, np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    )
    valid_values = current.copy()
    valid_values[1] = 0.1
    sim.SetJointValues('panda1', valid_values)
    current = sim.GetJointValues('panda1')
    np.testing.assert_allclose(
        current, np.array([0.0, 0.1, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    )

    # Make sure other robots didn't move.
    current = sim.GetJointValues('panda2')
    np.testing.assert_allclose(
        current, np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    )

  def test_no_collisions_at_start(self):
    sim = self.make_simulator()
    self.assertEmpty(sim.RobotsInCollision())

  def test_collision(self):
    sim = self.make_simulator()
    self.assertEmpty(sim.RobotsInCollision())
    sim.SetJointValues(
        'panda1', np.array(
            [0.0, -0.7854, -0.156768, -3.059768, 0.0, 1.396200, 0.0])
    )
    self.assertCountEqual(sim.RobotsInCollision(), ['panda1'])

  def test_robot_base_tcp_transforms(self):
    sim = self.make_simulator()
    panda1_base = sim.GetRobotBaseTransform('panda1')
    panda1_tcp = sim.GetRobotTcpTransform('panda1')
    self.assertSequenceEqual(panda1_base.shape, (4, 4))
    self.assertSequenceEqual(panda1_tcp.shape, (4, 4))
    np.testing.assert_allclose(
        panda1_base,
        np.array([
            [-1.0, 0.0, 0.0, 0.5515],
            [0.0, -1.0, 0.0, -0.6285],
            [0.0, 0.0, 1.0, 0.009],
            [0.0, 0.0, 0.0, 1.0],
        ]), atol=1e-3
    )
    np.testing.assert_allclose(
        panda1_tcp,
        np.array([
            [7.6597995e-01, 9.2599999e-05, 6.4286453e-01, 3.1327960e-01],
            [7.0935697e-05, -1.0000000e+00, 5.9522154e-05, -6.2849551e-01],
            [6.4286453e-01, 9.2670565e-09, -7.6597995e-01, 5.1928103e-01],
            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
        ]), atol=1e-3
    )

    panda2_base = sim.GetRobotBaseTransform('panda2')
    np.testing.assert_allclose(
        panda2_base,
        np.array([
            [1.0, 0.0, 0.0, -0.5515],
            [0.0, 1.0, 0.0, -0.6285],
            [0.0, 0.0, 1.0, 0.009],
            [0.0, 0.0, 0.0, 1.0],
        ]), atol=1e-3
    )

    panda1_base_to_tip = np.linalg.inv(panda1_base) @ panda1_tcp

    for robot in [f'panda{i}' for i in range(1, 5)]:
      base = sim.GetRobotBaseTransform(robot)
      tcp = sim.GetRobotTcpTransform(robot)
      base_to_tip = np.linalg.inv(base) @ tcp
      np.testing.assert_allclose(
          base_to_tip,
          panda1_base_to_tip,
          atol=1e-3,
      )

  def test_ik(self):
    sim = self.make_simulator()

    # This frame is in front of panda2 (at 0.0, -0.63, 1.0), pointing to +x.
    target_frame = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, -0.63],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    self.assertEmpty(sim.ComputeIk('panda1', target_frame, 1))
    self.assertEmpty(sim.ComputeIk('panda3', target_frame, 1))
    self.assertEmpty(sim.ComputeIk('panda4', target_frame, 1))

    panda2_solutions = sim.ComputeIk('panda2', target_frame, num_solutions=1)
    self.assertLen(panda2_solutions, 1)
    for solution in panda2_solutions:
      sim.SetJointValues('panda2', solution)
      self.assertEmpty(sim.RobotsInCollision())
      tcp = sim.GetRobotTcpTransform('panda2')
      translation_err = np.linalg.norm(tcp[:3, 3] - target_frame[:3, 3])
      self.assertLess(translation_err, 5e-2)

  def test_objects_and_transforms(self):
    sim = self.make_simulator()

    objects = sim.GetObjects()
    self.assertCountEqual(objects, [
        'workpiece', 'panda1', 'panda2', 'panda3', 'panda4'
    ])

    # For robots, object transform should be the same as the base transform.
    for robot in ['panda1', 'panda2', 'panda3', 'panda4']:
      np.testing.assert_allclose(
          sim.GetWorldTransform(robot),
          sim.GetRobotBaseTransform(robot)
      )

    workpiece_transform = sim.GetWorldTransform('workpiece')
    self.assertSequenceEqual(workpiece_transform.shape, (4, 4))
    np.testing.assert_allclose(
        workpiece_transform,
        np.eye(4, dtype=np.float32), atol=1e-3
    )

    sim.SetWorldTransform('workpiece', np.array([
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    workpiece_transform = sim.GetWorldTransform('workpiece')
    np.testing.assert_allclose(
        workpiece_transform,
        np.array([
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [-1.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]), atol=1e-3
    )

  def test_remove_objects(self):
    sim = self.make_simulator()

    self.assertCountEqual(sim.GetObjects(), [
        'workpiece', 'panda1', 'panda2', 'panda3', 'panda4'
    ])

    # The state of unrelated robots should not change after model recompile.
    sim.SetJointValues('panda4',
                       np.array([0.0, 0.1, 0.0, -2.1817, 0.0, 1.3962, 0.0]))
    panda4_tcp = sim.GetRobotTcpTransform('panda4')

    self.assertEqual(sim._model.nq, 4 * 7)
    sim.RemoveObject('panda3')

    np.testing.assert_allclose(
        sim.GetJointValues('panda4'),
        np.array([0.0, 0.1, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    )
    np.testing.assert_allclose(
        sim.GetRobotTcpTransform('panda4'),
        panda4_tcp
    )

    self.assertCountEqual(sim.GetObjects(), [
        'workpiece', 'panda1', 'panda2', 'panda4'
    ])
    sim.RemoveObject('workpiece')
    self.assertCountEqual(sim.GetObjects(), [
        'panda1', 'panda2', 'panda4'
    ])

  def test_add_box(self):
    sim = self.make_simulator()

    self.assertCountEqual(sim.GetObjects(), [
        'workpiece', 'panda1', 'panda2', 'panda3', 'panda4'
    ])

    box_id = sim.AddBox(np.array([1.0, 2.0, 3.0]))
    self.assertEqual(box_id, 'box_0')
    self.assertCountEqual(sim.GetObjects(), [
        'workpiece', 'panda1', 'panda2', 'panda3', 'panda4', 'box_0'
    ])

    box_id = sim.AddBox(np.array([2.0, 3.0, 4.0]))
    self.assertEqual(box_id, 'box_1')
    self.assertCountEqual(sim.GetObjects(), [
        'workpiece', 'panda1', 'panda2', 'panda3', 'panda4', 'box_0', 'box_1'
    ])

    box_id = sim.AddBox(np.array([1.0, 2.0, 3.0]), prefix='my_box_')
    self.assertEqual(box_id, 'my_box_2')
    self.assertCountEqual(sim.GetObjects(), [
        'workpiece', 'panda1', 'panda2', 'panda3', 'panda4', 'box_0', 'box_1',
        'my_box_2'
    ])

  def test_meshes(self):
    sim = self.make_simulator()

    panda1_meshes = sim.GetMeshes('panda1')
    self.assertLen(panda1_meshes, 15)

    for mesh in panda1_meshes:
      face_vertices_max = np.max(mesh.faces)
      face_vertices_min = np.min(mesh.faces)
      self.assertEqual(face_vertices_min, 0)
      self.assertEqual(face_vertices_max, mesh.vertices.shape[0] - 1)

    all_vertices = np.concatenate([
        mesh.vertices for mesh in sim.GetMeshes('panda1')])
    self.assertSequenceEqual(all_vertices.shape, (1359, 3))
    vertices_mean = np.mean(all_vertices, axis=0)
    np.testing.assert_allclose(
        vertices_mean,
        np.array([0.484038, -0.633638, 0.545283]),
        atol=1e-3
    )

    all_vertices = np.concatenate([
        mesh.vertices for mesh in sim.GetMeshes('panda3')])
    self.assertSequenceEqual(all_vertices.shape, (1359, 3))
    vertices_mean = np.mean(all_vertices, axis=0)
    np.testing.assert_allclose(
        vertices_mean,
        np.array([-0.484038, 0.633638, 0.545283]),
        atol=1e-3
    )

    box_id = sim.AddBox(np.array([1.0, 2.0, 3.0]))
    transform = np.array([
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    sim.SetWorldTransform(box_id, transform)
    meshes = sim.GetMeshes(box_id)
    self.assertLen(meshes, 1)
    self.assertEqual(meshes[0].vertices.shape, (8, 3))
    self.assertEqual(meshes[0].faces.shape, (12, 3))
    vertices_mean = np.mean(meshes[0].vertices, axis=0)
    np.testing.assert_allclose(
        vertices_mean, np.array([4.0, 5.0, 6.0]), atol=1e-3
    )

  def test_copy(self):
    sim = self.make_simulator()
    old_copy_joint_values = np.array([0.0, 0.1, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    new_copy_joint_values = np.array([0.1, 0.2, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    sim.SetJointValues('panda1', old_copy_joint_values)
    sim_copy = copy.copy(sim)
    np.testing.assert_allclose(
        sim_copy.GetJointValues('panda1'), old_copy_joint_values
    )
    sim_copy.SetJointValues('panda1', new_copy_joint_values)
    np.testing.assert_allclose(
        sim_copy.GetJointValues('panda1'), new_copy_joint_values
    )
    # The original should remain unchanged
    np.testing.assert_allclose(
        sim.GetJointValues('panda1'), old_copy_joint_values
    )

  def test_remove_object_bug_repro(self):
    sim_original = self.make_simulator()
    sim = copy.copy(sim_original)
    expected_joint_values = np.array([
        0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0])
    np.testing.assert_allclose(
        sim.GetJointValues('panda1'), expected_joint_values
    )
    sim.RemoveObject('workpiece')
    np.testing.assert_allclose(
        sim.GetJointValues('panda1'), expected_joint_values
    )


if __name__ == '__main__':
  absltest.main()
