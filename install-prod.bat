@echo off
REM Production installation script for Windows
REM Installs CPU-only PyTorch (much smaller than CUDA version)

echo Installing CPU-only PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing other dependencies...
pip install -r requirements-prod.txt

echo Production dependencies installed!

