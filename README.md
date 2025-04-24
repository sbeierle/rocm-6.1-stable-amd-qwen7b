# ROCm 6.1 Qwen Setup (Stable)

This repository documents a complete setup of ROCm 6.1 with Qwen inference using AMD GPUs (tested with RX 7900 XTX). It includes:

- Ubuntu 22.04 + Kernel 6.8 / 6.11
- Python 3.10 / 3.12 compatible
- torch >= 2.1.0 (ROCm nightly) + safetensors OK
- `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION` etc.
- `torch.cuda.is_available()` = ✅ True on AMD!
- `safetensors`, `transformers`, `huggingface_hub`
- Testen mit: `python -c "import torch; print(torch.cuda.is_available())"`

## Setup Overview

```bash
sudo apt update
sudo apt install wget gnupg2 git nano micro
```

### Add AMD ROCm Repository

```bash
wget https://repo.radeon.com/amdgpu-install/6.1.0/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb
sudo apt install ./amdgpu-install_6.1.60100-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
```

### Set Environment Variables

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## Test ROCm
```bash
rocminfo
/opt/rocm/bin/hipconfig
/opt/rocm/bin/rocminfo | grep gfx
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Example Shell

See: `scripts/start_qwen_env.sh`
# rocm-6.1-qwen
