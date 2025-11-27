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

"""Tests for the geometry library."""

from absl.testing import absltest
import numpy as np
from scipy.spatial import transform
from data import data_locator
from environments import geometry
from environments import mujoco_simulator


class BaseEnvTest(absltest.TestCase):

  def test_sample_array(self):
    # Mismatched shapes.
    with self.assertRaisesRegex(ValueError, 'must have the same shape'):
      geometry.sample_array(
          mins=np.array([0.1, 0.2, 0.3]), maxs=np.array([0.2, 0.3, 0.4, 0.5])
      )

    # Invalid range.
    with self.assertRaisesRegex(
        ValueError, 'mins\\[i\\] must be < maxs\\[i\\]'
    ):
      geometry.sample_array(
          mins=np.array([0.1, 0.5, 0.3]), maxs=np.array([0.2, 0.3, 0.4])
      )

    mins = np.array([10.0, 10.0, 30.0])
    maxs = np.array([20.0, 20.0, 40.0])
    sample = geometry.sample_array(mins=mins, maxs=maxs)
    self.assertTrue((sample >= mins).all())
    self.assertTrue((sample <= maxs).all())

    # We should be drawing independent samples, so we check [0] and [1] are not
    # the same. This is theoretically flaky if we happen to get the same
    # mantissas (~1/2^52 for double precision floats).
    self.assertNotEqual(sample[0], sample[1])

  def test_sample_rotation(self):
    sampled_rotation = geometry.sample_rotation()
    self.assertSequenceEqual(sampled_rotation.shape, (3, 3))

    # A rotation matrix must be orthogonal (transpose equals inverse) with
    # determinant == 1.
    np.testing.assert_allclose(
        np.transpose(sampled_rotation), np.linalg.inv(sampled_rotation)
    )
    self.assertAlmostEqual(np.linalg.det(sampled_rotation), 1.0)

    # We are astronomically unlikely to get two identical rotations.
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(sampled_rotation, geometry.sample_rotation())

  def test_normalize(self):
    # Smaller than epsilon.
    np.testing.assert_allclose(
        geometry.normalize(np.array([0.1, 0.1, 0.1]), epsilon=0.5),
        np.array([0.2, 0.2, 0.2]),
    )
    # Normal case.
    np.testing.assert_allclose(
        geometry.normalize(np.array([2.0, 5.0, 3.0])),
        np.array([0.324443, 0.811107, 0.486664]),
        atol=0.001,
    )

  def test_calculate_triangle_properties(self):
    # Triangle at (1, 0, 0), (0, 1, 0), (0, 0, 1)
    properties = geometry.calculate_triangle_properties(np.eye(3))
    self.assertAlmostEqual(properties.area, 0.8660254037844386)
    np.testing.assert_allclose(
        properties.centroid, np.array([0.333, 0.333, 0.333]), atol=0.001
    )

    # There are 2 possible normal directions. We enforce this normal direction
    # as it results in an outward normal for triangles in meshes with CCW
    # winding order (convention for OpenGL and DirectX, so most meshes are
    # specified this way).
    np.testing.assert_allclose(
        properties.normal, np.array([0.57735, 0.57735, 0.57735]), atol=0.001
    )

    # Degenerate triangle.
    degenerate_triangle = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    properties = geometry.calculate_triangle_properties(degenerate_triangle)
    self.assertAlmostEqual(properties.area, 0.0)
    np.testing.assert_allclose(
        properties.centroid, np.array([0.667, 0.0, 0.0]), atol=0.001
    )
    self.assertAlmostEqual(
        np.dot(properties.normal, np.array([1.0, 0.0, 0.0])), 0.0
    )

  def test_sample_points_from_triangle(self):
    triangle = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    for _ in range(100):
      point = geometry.sample_point_from_triangle(triangle)
      self.assertAlmostEqual(point[2], 1.0)
      self.assertGreaterEqual(point[0], 1.0)
      self.assertLessEqual(point[0], 2.0)
      self.assertGreaterEqual(point[1], 1.0)
      self.assertLessEqual(point[1], 2.0)
      self.assertLessEqual(point[0] + point[1], 3.0)

  def test_sample_frame_with_z_close_to(self):
    vz = np.array([1.0, 2.0, 3.0])
    f = geometry.sample_frame_with_z_close_to(vz, z_tol=(np.pi / 4.0))
    self.assertLessEqual(
        geometry.angle_between_vectors_rad(vz, f[:, 2]), np.pi / 4.0
    )
    self.assertAlmostEqual(np.dot(f[:, 0], f[:, 1]), 0.0)
    self.assertAlmostEqual(np.dot(f[:, 0], f[:, 2]), 0.0)
    self.assertAlmostEqual(np.dot(f[:, 1], f[:, 2]), 0.0)

  def test_angle_between_vectors(self):
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    self.assertAlmostEqual(
        geometry.angle_between_vectors_rad(v1, v2), np.pi / 4.0
    )
    v2 = np.array([1.0, 0.0, 0.0])
    self.assertAlmostEqual(geometry.angle_between_vectors_rad(v1, v2), 0.0)

  def test_convex_decomposition(self):
    full_path = data_locator.get_data_path(
        'mujoco_world/4_pandas_world.xml')
    config = mujoco_simulator.MujocoWorldConfig(
        xml_path=full_path,
    )
    world = mujoco_simulator.MujocoWorld(config)
    self.assertIn('workpiece', world.GetObjects())
    workpiece_meshes = world.GetMeshes('workpiece')
    self.assertLen(workpiece_meshes, 1)
    mesh = workpiece_meshes[0]
    decomp = geometry.convex_decomposition(
        vertices=mesh.vertices, faces=mesh.faces, max_parts=24
    )
    self.assertLen(decomp, 24)

  def test_bounding_boxes(self):
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    aabb = geometry.aabb_from_points(vertices)
    self.assertAlmostEqual(aabb.volume(), 1.0)
    np.testing.assert_allclose(aabb.spans, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(aabb.pose[:3, 3], [0.5, 0.5, 0.5])
    obb = geometry.obb_from_points(vertices)
    self.assertAlmostEqual(obb.volume(), 1.0, places=2)
    np.testing.assert_allclose(obb.spans, [1.0, 1.0, 1.0], atol=1e-3)
    np.testing.assert_allclose(obb.pose[:3, 3], [0.5, 0.5, 0.5], atol=1e-3)

    # Should be translationally-invariant.
    translated = vertices + np.array([1.0, 2.0, 3.0])
    aabb = geometry.aabb_from_points(translated)
    self.assertAlmostEqual(aabb.volume(), 1.0)
    np.testing.assert_allclose(aabb.spans, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(aabb.pose[:3, 3], [1.5, 2.5, 3.5])
    obb = geometry.obb_from_points(translated)
    self.assertAlmostEqual(obb.volume(), 1.0, places=2)
    np.testing.assert_allclose(obb.pose[:3, 3], [1.5, 2.5, 3.5], atol=1e-3)

    # Apply a rotation.
    rotation = transform.Rotation.from_euler('xyz', [45, 45, 45], degrees=True)
    rotated = rotation.apply(vertices)
    aabb = geometry.aabb_from_points(rotated)
    self.assertAlmostEqual(aabb.volume(), 3.84099, places=3)
    np.testing.assert_allclose(aabb.spans, [1.5, 1.5, 1.707107], atol=1e-3)
    np.testing.assert_allclose(
        aabb.pose[:3, 3], [0.603553, 0.603553, 0.146447], atol=1e-3
    )
    obb = geometry.obb_from_points(rotated)
    self.assertAlmostEqual(obb.volume(), 1.0, places=2)
    np.testing.assert_allclose(
        obb.pose[:3, 3], [0.603553, 0.603553, 0.146447], atol=1e-3
    )


if __name__ == '__main__':
  absltest.main()
