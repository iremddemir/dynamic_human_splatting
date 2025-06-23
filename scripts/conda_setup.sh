#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

#!/bin/bash
eval "$(conda shell.bash hook)"
conda create -n hugs310 python=3.10 -y

conda activate hugs310

conda install -y pytorch=2.1.2 \
  torchvision=0.16.2 \
  torchaudio=2.1.2 \
  pytorch-cuda=12.1 \
  -c pytorch -c nvidia

pip install fvcore iopath
conda install pytorch3d -c conda-forge

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

pip install -r requirements.txt
pip install git+https://github.com/mattloper/chumpy.git
