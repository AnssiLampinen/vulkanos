#!/usr/bin/env bash
# Make .venv with CUDA torch and the rest of requirements.txt, then check the GPU.
#   bash setup.sh            # CUDA 12.6 wheels, needs nvidia driver >= 560 (nvidia-smi shows it)
#   CUDA=cu118 bash setup.sh # older drivers (>= 450)
#   CUDA=cpu bash setup.sh   # no GPU
set -e
cd "$(dirname "$0")"
CUDA=${CUDA:-cu126}
PY=${PY:-python3}
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "now: . .venv/bin/activate && python scripts/gpu_run.py"
