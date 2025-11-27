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

"""Utility functions."""

from collections.abc import Mapping

import numpy as np

from environments import data_pb2


ArrayTree = Mapping[str, 'ArrayTree | np.ndarray']


def to_proto(data: np.ndarray) -> data_pb2.NDArray:
  """Converts a numpy array to a proto."""
  ret = data_pb2.NDArray()
  ret.dims.extend(data.shape)
  ret.dtype = data.dtype.str
  flat_data = data.flatten().tolist()
  if data.dtype == np.float32:
    ret.float_values.extend(flat_data)
  elif data.dtype == np.float64:
    ret.double_values.extend(flat_data)
  elif (
      data.dtype == np.int8
      or data.dtype == np.uint8
      or data.dtype == np.int16
      or data.dtype == np.uint16
      or data.dtype == np.int32
      or data.dtype == np.uint32
      or data.dtype == np.int64
  ):
    ret.int64_values.extend(flat_data)
  else:
    raise ValueError(f'Unsupported dtype: {data.dtype}')
  return ret


def from_proto(proto: data_pb2.NDArray) -> np.ndarray:
  """Reconstructs a numpy array from a proto."""
  dtype = np.dtype(proto.dtype)
  if dtype == np.float32:
    values = proto.float_values
  elif dtype == np.float64:
    values = proto.double_values
  elif (
      dtype == np.int8
      or dtype == np.uint8
      or dtype == np.int16
      or dtype == np.uint16
      or dtype == np.int32
      or dtype == np.uint32
      or dtype == np.int64
  ):
    values = proto.int64_values
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')
  return np.array(values, dtype=dtype).reshape(proto.dims)


def pytree_to_proto(t: ArrayTree) -> data_pb2.PyTree:
  """Converts a PyTree to a proto."""
  proto = data_pb2.PyTree()
  for key, value in t.items():
    proto.keys.append(key)
    if isinstance(value, np.ndarray) or isinstance(value, np.number):
      proto.entries.append(data_pb2.PyTreeEntry(value=to_proto(value)))
    elif isinstance(value, dict):
      proto.entries.append(data_pb2.PyTreeEntry(subtree=pytree_to_proto(value)))
    else:
      raise ValueError(f'Unsupported value type: {type(value)}')
  return proto


def pytree_from_proto(proto: data_pb2.PyTree) -> ArrayTree:
  """Reconstructs a PyTree from a proto."""
  ret = {}
  for key, entry in zip(proto.keys, proto.entries):
    if entry.HasField('value'):
      ret[key] = from_proto(entry.value)
    elif entry.HasField('subtree'):
      ret[key] = pytree_from_proto(entry.subtree)
  return ret
