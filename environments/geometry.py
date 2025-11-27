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

"""Coordinate geometry library."""

from collections.abc import Sequence
import dataclasses
import math

import numpy as np
from scipy.spatial import transform
import vhacdx


_EPSILON = 1e-8


@dataclasses.dataclass(frozen=True)
class BoundingBox:
  """An axis-aligned or oriented bounding box."""
  spans: np.ndarray  # (3,)
  pose: np.ndarray  # (4, 4)

  def volume(self) -> float:
    return np.prod(self.spans)


def sample_array(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
  """Uniformly samples an array between mins and maxs."""
  if mins.shape != maxs.shape:
    raise ValueError(
        f'mins and maxs must have the same shape: {mins.shape} vs {maxs.shape}'
    )
  if not (mins < maxs).all():
    raise ValueError(
        f'mins[i] must be < maxs[i] for all i: {mins} vs {maxs}'
    )
  return (maxs - mins) * np.random.random_sample(mins.shape) + mins


def sample_rotation() -> np.ndarray:
  """Uniformly samples a rotation (uniform on sphere) as a rotation matrix.

  Returns:
    Rotation matrix as a 3x3 np.ndarray.
  """
  # A quaternion with all components drawn from a normal distribution will give
  # us a uniformly distributed rotation (Graphics Gems 3: III.6). Handle the
  # degenerate case separately.
  rotation_quat = None
  while rotation_quat is None or np.linalg.norm(rotation_quat) < 0.0001:
    rotation_quat = np.random.normal(size=(4,))
  return transform.Rotation.from_quat(rotation_quat.tolist()).as_matrix()


def sample_point_from_triangle(triangle: np.ndarray) -> np.ndarray:
  """Samples a point from a triangle defined by vertices."""
  # First we take the vectors AB and AC.
  a_to_b = triangle[1, :] - triangle[0, :]
  a_to_c = triangle[2, :] - triangle[0, :]
  # Then we draw two numbers in [0, 1] to create a lienar combination of the two
  # vectors. This gives us a point inside the parallelogram formed by AB + AC
  # and AC + AB.
  coeff_1 = np.random.random_sample()
  coeff_2 = np.random.random_sample()
  # If they add up to > 1.0, our new point is in the half of the parallelogram
  # that's not part of the triangle, so we reflect that back into the triangle.
  if (coeff_1 + coeff_2) > 1.0:
    coeff_1 = 1.0 - coeff_1
    coeff_2 = 1.0 - coeff_2
  return a_to_b * coeff_1 + a_to_c * coeff_2 + triangle[0, :]


def normalize(v: np.ndarray, epsilon: float = _EPSILON) -> np.ndarray:
  """Normalizes a vector."""
  return v / max(np.linalg.norm(v), epsilon)


@dataclasses.dataclass
class TriangleProperties:
  """Properties of a triangle.

  Attributes:
    centroid: Centroid of the triangle.
    normal: Outward-facing normal direction based on CCW winding order.
    area: Area of the triangle.
  """
  centroid: np.ndarray  # (3,)
  normal: np.ndarray  # (3,)
  area: float


def calculate_triangle_properties(triangle: np.ndarray) -> TriangleProperties:
  """Calculates TriangleProperties for a triangle specified in vertices."""
  ab = triangle[1, :] - triangle[0, :]
  ac = triangle[2, :] - triangle[0, :]
  cross = np.cross(ab, ac)
  return TriangleProperties(
      centroid=np.mean(triangle, axis=0),
      normal=normalize(cross),
      area=0.5 * np.linalg.norm(cross),
  )


def sample_frame_with_z_close_to(z: np.ndarray, z_tol: float) -> np.ndarray:
  """Samples a rotation matrix where the +z axis is less than z_tol from z."""
  # This may be a very inefficient way of doing it, but should be good enough
  # for our usecase, esp since we are using a high z_tol.
  while True:
    candidate = sample_rotation()
    if angle_between_vectors_rad(candidate[:, 2], z) < z_tol:
      return candidate


def angle_between_vectors_rad(v1: np.ndarray, v2: np.ndarray) -> float:
  """Returns the angle between two vectors in radians."""
  cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
  return math.acos(np.clip(cos_theta, -1.0, 1.0))


def convex_decomposition(
    vertices: np.ndarray, faces: np.ndarray, max_parts: int = 64,
    percent_volume_err_tol: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
  """Decomposes a surface mesh into a set of convex polyhedrons."""

  # vhacdx requires the mesh to be in a very strange format, where the faces
  # are flattened, each specified by the number of vertices (always 3) and the
  # indices of the vertices, even though it (and vhacd) assumes the faces are
  # triangles anyways.
  vertex_counts = 3 * np.ones(shape=(faces.shape[0], 1), dtype=np.int32)
  faces_with_counts = np.concatenate([vertex_counts, faces], axis=1).flatten()

  # This computes 2^maxRecursionDepth convex hulls, then merge them to produce
  # maxConvexHulls.
  return vhacdx.compute_vhacd(
      points=vertices,
      faces=faces_with_counts,
      maxConvexHulls=max_parts,
      minimumVolumePercentErrorAllowed=percent_volume_err_tol,
      maxRecursionDepth=8,
      findBestPlane=True,
      asyncACD=False,
  )


def aabb_from_points(
    vertices: np.ndarray) -> BoundingBox:
  """Returns the axis-aligned bounding box of a set of points."""
  mins = np.min(vertices, axis=0)
  maxs = np.max(vertices, axis=0)
  spans = maxs - mins
  center = mins + spans / 2.0
  pose = np.eye(4)
  pose[:3, 3] = center
  return BoundingBox(spans=spans, pose=pose)


# The best known exact optimal oriented bounding box algorithm is O(N^3) time
# (O’Rourke, 1985), which is too slow for practical use.
# There are several approximations. State of the art approach seems to be
# global optimization in the space of 3D rotation matrices based on the paper
# "Fast oriented bounding box optimization on the rotation group SO(3, R)" by
# Chia-Tche Chang, Bastien Gorissen, and Samuel Melchior.
# This is an implementation of the proposed algorithm.
class SimplexVertex:
  """A vertex in a simplex.

  In Nelder-Mead, a simplex consists of 4 rotation matrices, each with its own
  fitness (negated volume of the AABB of the vertices after rotating using the
  matrix).
  """

  def __init__(self, rotation_matrix: np.ndarray | None, vertices: np.ndarray):
    if rotation_matrix is None:
      rotation_matrix = sample_rotation()
    self._matrix = rotation_matrix

    # The fitness of this rotation matrix is the negated volume of the AABB of
    # the vertices after rotating using the matrix.
    rotated_vertices = np.matmul(rotation_matrix, vertices.T)
    self._fitness = -np.prod(np.max(rotated_vertices, axis=1) -
                             np.min(rotated_vertices, axis=1))

  def matrix(self) -> np.ndarray:
    return self._matrix

  def fitness(self) -> float:
    return self._fitness


def _simplex_fitness(simplex: list[SimplexVertex]) -> float:
  """Returns the fitness of the simplex (best fitness of the 4 matrices)."""
  return max([v.fitness() for v in simplex])


def _centroid(simplex_vertices: list[SimplexVertex],
              vertices: np.ndarray) -> SimplexVertex:
  """Returns the centroid of the simplex vertices."""
  mean = sum([v.matrix() for v in simplex_vertices]) / len(simplex_vertices)
  centroid = np.linalg.qr(mean).Q
  return SimplexVertex(centroid, vertices)


def _nelder_mead_step(simplex: Sequence[SimplexVertex],
                      vertices: np.ndarray) -> list[SimplexVertex]:
  """Performs one Nelder-Mead step on the simplex."""
  # Sort by fitness.
  simplex = sorted(simplex, key=lambda x: x.fitness(), reverse=True)
  # Now we replace the worst matrix with its reflection over the centroid of the
  # other 3 matrices.
  centroid = _centroid(simplex[:3], vertices)

  worst = simplex[3]

  reflection = np.matmul(centroid.matrix(), worst.matrix().T)
  reflection = np.matmul(reflection, centroid.matrix())
  reflection = SimplexVertex(reflection, vertices)

  # If the reflection is better than the next worst, we accept it.
  next_worst_fitness = simplex[2].fitness()
  best_fitness = simplex[0].fitness()
  if reflection.fitness() > next_worst_fitness:
    if reflection.fitness() > best_fitness:
      simplex[3] = reflection
    else:
      # The reflection is better than the next worst, but not the best.
      # Try expanding the reflection.
      expansion = np.matmul(centroid.matrix(), worst.matrix().T)
      expansion = np.matmul(expansion, reflection.matrix())
      expansion = SimplexVertex(expansion, vertices)
      # If the expansion is better, replace the worst matrix with it, otherwise
      # use the reflection.
      if expansion.fitness() > reflection.fitness():
        simplex[3] = expansion
      else:
        simplex[3] = reflection
  else:
    # The reflection is worse than the next worst, so we try contraction.
    mean = _centroid([centroid, worst], vertices)
    # If mean is better than the worst, replace the worst with the mean.
    if mean.fitness() > worst.fitness():
      simplex[3] = mean
    else:
      # Otherwise, move the whole simplex towards the best.
      for i in range(4):
        simplex[i] = _centroid([simplex[i], simplex[0]], vertices)
  return simplex


def _get_best_vertex(
    population: Sequence[Sequence[SimplexVertex]]) -> SimplexVertex:
  """Returns the best vertex in the population."""
  best = population[0][0]
  for simplex in population:
    for vertex in simplex:
      if vertex.fitness() > best.fitness():
        best = vertex
  return best


def obb_from_points(vertices: np.ndarray,
                    pop_size: int = 20, nelder_mead_iterations: int = 25,
                    max_iterations: int = 100) -> BoundingBox:
  """Returns the oriented bounding box of a set of points."""
  assert pop_size % 2 == 0, 'pop_size must be even'

  # As with most OBB algorithms, a good pre-processing step is to calculate the
  # convex hull of the vertices. We skip that here because our input is from
  # V-HACD, which already gives us convex hulls.

  # First, create a set of pop_size (M) simplices each with 4 rotation matrices.
  population = [[SimplexVertex(rotation_matrix=None,
                               vertices=vertices)
                 for _ in range(4)] for _ in range(pop_size)]

  iterations_without_improvement = 0
  best = population[0][0]
  iteration = 0

  while iterations_without_improvement < 5 and iteration < max_iterations:
    # For each simplex, calculate the fitness (minimum volume using the
    # matrices in the simplex).
    population_with_fitnesses = [
        (simplex, _simplex_fitness(simplex)) for simplex in population
    ]

    # Drop pop_size/2 worst simplexes by fitness.
    population_with_fitnesses = sorted(
        population_with_fitnesses, key=lambda x: x[1], reverse=True)
    half_population = [
        simplex for simplex, _ in population_with_fitnesses[:pop_size // 2]]

    # Create the two crossover populations. The first one is a random mix of
    # the two parents, but higher fitness parents are more likely to be chosen.
    crossover_i = []
    for _ in range(pop_size // 2):
      parent_simplexes = [
          half_population[np.random.randint(len(half_population))],
          half_population[np.random.randint(len(half_population))]
      ]
      crossover_simplex = []
      for j in range(4):
        parent0_better = (parent_simplexes[0][j].fitness() >
                          parent_simplexes[1][j].fitness())
        prob_parent0 = 0.6 if parent0_better else 0.4
        if np.random.random_sample() < prob_parent0:
          crossover_simplex.append(parent_simplexes[0][j])
        else:
          crossover_simplex.append(parent_simplexes[1][j])
      crossover_i.append(crossover_simplex)

    # Second population by weighted combination.
    crossover_ii = []
    for _ in range(pop_size // 2):
      parent_simplexes = [
          half_population[np.random.randint(len(half_population))],
          half_population[np.random.randint(len(half_population))]
      ]
      crossover_simplex = []
      for j in range(4):
        parent0_better = (parent_simplexes[0][j].fitness() >
                          parent_simplexes[1][j].fitness())
        weight_parent0 = 0.6 if parent0_better else 0.4
        mixed_matrix = weight_parent0 * parent_simplexes[0][j].matrix() + (
            1.0 - weight_parent0) * parent_simplexes[1][j].matrix()
        # QR factorization is needed to ensure the matrix is still a rotation.
        mixed_matrix = np.linalg.qr(mixed_matrix).Q
        crossover_simplex.append(SimplexVertex(rotation_matrix=mixed_matrix,
                                               vertices=vertices))
      crossover_ii.append(crossover_simplex)

    # The new population is the two new crossover groups.
    population = crossover_i + crossover_ii
    assert len(population) == pop_size

    # Run nm_iteration Nelder-Mead iterations.
    for i in range(pop_size):
      for _ in range(nelder_mead_iterations):
        population[i] = _nelder_mead_step(population[i], vertices)

    new_best = _get_best_vertex(population)
    if new_best.fitness() <= best.fitness():
      iterations_without_improvement += 1
    else:
      iterations_without_improvement = 0
      best = new_best
    iteration += 1
  best_matrix = best.matrix()
  rotated_vertices = np.matmul(best_matrix, vertices.T)
  mins = np.min(rotated_vertices, axis=1)
  maxs = np.max(rotated_vertices, axis=1)
  spans = maxs - mins
  center = mins + spans / 2.0

  # The center is in the OBB frame, so we need to transform it back to the
  # world frame. For a rotation matrix, the inverse is just the transpose.
  center = np.matmul(best_matrix.T, center)

  pose = np.eye(4)
  pose[:3, 3] = center
  pose[:3, :3] = best_matrix
  return BoundingBox(
      spans=spans.astype(np.float32), pose=pose.astype(np.float32)
  )
