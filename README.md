# RoboBallet

This is the code release accompanying our paper "RoboBallet: Planning for
Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning".

## Installation

```
# Create a virtual environment.
# Python 3.10 must be used due to the various dependencies all have strict
# Python version requirements. If not available from your system repository,
# use pyenv.
python3.10 -m venv venv
source venv/bin/activate
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install the right NVIDIA cuDNN runtime libraries.
pip install nvidia-cudnn-cu12==8.9.4.*

# Install rest of the dependencies.
pip install -r requirements.txt

# We are using an older version of jaxlib where dynamic libraries have the
# executable stack bit set (as was the default when they were built).
# glibc 2.41+ no longer allows that. They don't actually execute the stack, so
# we can just clear the flag.
# See: https://github.com/jax-ml/jax/issues/26781
find venv/lib/python3.10/site-packages/jaxlib -name '*.so' -exec execstack -c {} \;

# Install protobuf compiler and generate accessor library for our protobufs.
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protoc-3.19.4-linux-x86_64.zip
unzip protoc-3.19.4-linux-x86_64.zip -d /tmp
/tmp/bin/protoc --proto_path=roboballet --python_out=roboballet roboballet/environments/data.proto
```

## Usage
```
python -m roboballet.launch.launch --lp_launch_type=local_mp
```

This starts training with model checkpoints saved to ./model_checkpoints, and
tensorboard log files with metrics to ./tensorboard_log.

## Citing this work

```
@article{lai2025roboballet,
  title={RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning},
  author={Lai, Matthew and Go, Keegan and Li, Zhibin and Kroger, Torsten and Schaal, Stefan and Allen, Kelsey and Scholz, Jonathan},
  journal={Science Robotics},
  volume={10},
  number={106},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
```

## Known issues

* During the training process, we use an IK solver to filter for tasks that at
  least have one IK solution. We have included pre-generated start states so
  that our results can be reproduced, but for adapting this codebase for future
  work, an IK implementation needs to be provided.
  See ik_solver.py.

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
