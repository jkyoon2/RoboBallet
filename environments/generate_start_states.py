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

"""Script to pre-generate start states for PlanningEnv."""

import multiprocessing as mp
from multiprocessing import sharedctypes
import os
from typing import Sequence

from absl import app
from absl import flags
import tqdm

from environments import data_pb2
from environments import planning_env
from launch import hparams


_NUM_TARGETS = flags.DEFINE_multi_integer(
    'num_targets', [10, 40], 'Number of targets to generate.'
)

_NUM_OBSTACLES = flags.DEFINE_multi_integer(
    'num_obstacles', [30], 'Number of obstacles to generate.'
)

_NUM_ROBOTS = flags.DEFINE_multi_integer(
    'num_robots', [4, 8], 'Number of robots in the environment.'
)

_NUM_STATES_TO_GENERATE = flags.DEFINE_integer(
    'num_states_to_generate', 50000, 'Number of start states to generate.'
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/start_states',
    'Path to the output file.',
)

_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 0, 'Number of workers to use. 0 for CPU count.'
)


def _generate_worker(
    configs: hparams.LaunchConfigs,
    requests_remaining: 'sharedctypes.Synchronized[int]',
    output_queue: mp.Queue):
  """Generate worker."""
  while True:
    with requests_remaining.get_lock():
      if requests_remaining.value <= 0:
        print('Worker exiting because queue is empty.')
        break
      else:
        requests_remaining.value -= 1

    try:
      env = planning_env.PlanningEnv(config=configs.env_config)
      env.reset()
      start_state = env.get_start_state()
    except Exception as e:  # pylint: disable=broad-exception-caught
      output_queue.put(e)
      continue
    output_queue.put(start_state.to_proto())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  if _NUM_WORKERS.value == 0:
    num_workers = mp.cpu_count()
  else:
    num_workers = _NUM_WORKERS.value

  print(f'Using {num_workers} workers.')

  for num_targets in _NUM_TARGETS.value:
    for num_obstacles in _NUM_OBSTACLES.value:
      for num_robots in _NUM_ROBOTS.value:
        print(f'Generating start states for {num_targets} targets, '
              f'{num_obstacles} obstacles, {num_robots} robots.')
        config = hparams.make_optimised_configs(
            num_targets=num_targets,
            num_obstacles=num_obstacles,
            num_robots=num_robots,
        )

        requests_remaining = mp.Value('i', _NUM_STATES_TO_GENERATE.value,
                                      lock=True)

        output_queue = mp.Queue(maxsize=num_workers)

        processes = []
        for _ in range(num_workers):
          process = mp.Process(
              target=_generate_worker, args=(config, requests_remaining,
                                             output_queue))
          process.start()
          processes.append(process)

        out_proto = data_pb2.StartStates()

        for _ in tqdm.tqdm(range(_NUM_STATES_TO_GENERATE.value)):
          output_item = output_queue.get()
          if isinstance(output_item, Exception):
            raise output_item
          out_proto.start_states.append(output_item)

        for process in processes:
          process.join()

        output_path = os.path.join(
            _OUTPUT_DIR.value,
            f'{num_robots}r_{num_targets}'
            f't_{num_obstacles}o.pb',
        )

        with gfile.Open(output_path, 'wb') as f:
          f.write(out_proto.SerializeToString())


if __name__ == '__main__':
  mp.set_start_method('forkserver')
  app.run(main)
