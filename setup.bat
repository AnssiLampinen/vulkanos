@echo off
rem Make .venv with CUDA torch and the rest of requirements.txt, then check the GPU.
rem   setup.bat           CUDA 12.6 wheels, needs nvidia driver >= 560 (nvidia-smi shows it)
rem   setup.bat cu118     older drivers
rem   setup.bat cpu       no GPU
cd /d "%~dp0"
set CUDA=%1
if "%CUDA%"=="" set CUDA=cu126
if not exist .venv python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/%CUDA% || exit /b 1
pip install -r requirements.txt || exit /b 1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo now: .venv\Scripts\activate.bat ^&^& python scripts\gpu_run.py
