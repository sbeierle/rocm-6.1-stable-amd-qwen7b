# 🔧 ROCm 6.1 Qwen Setup (AMD RX 7900 | Torch | Safetensors)

> ✅ Fully working ROCm + Torch + Safetensors Setup for Qwen Inference  
> 🧠 Focus: Ethical LLM Reverse Engineering & Red Teaming  
> ⚙️ GPU: AMD Radeon RX 7900 XTX | ROCm 6.1 | Python 3.10 & 3.12 | Linux Kernel 6.8 / 6.11 | Ubuntu 22.04 & 24.10

---

## 📦 Project Description

This repository documents a complete and tested AMD ROCm 6.1 environment for running Qwen-based LLMs using Safetensors.  
It's part of an ethical research project on LLM deconstruction, bias analysis, and Red Teaming using AMD hardware.

This is **not just a proof-of-concept** – it's a **reproducible and hardened setup**, including:

- 🛠️ CUDA-free Torch inference on AMD GPUs (HIP)
- 📁 Full Safetensors support via `AutoModelForCausalLM`
- ⚡ Live interactive GPU inference (`interactive4.py`)
- 🔍 Tools for bias, RLHF and vocabulary inspection (see `qwen-project`)

### Tested Environments:
- **Ubuntu 22.04** with Python **3.10** & Kernel **6.8**
- **Ubuntu 24.10** with Python **3.12** & Kernel **6.11.x**
- On **Ubuntu 24.10**, newer `transformers` and `numpy>=2.0` were confirmed working.

---

## ✅ Features

- ROCm 6.1 + HIP + Torch fully integrated
- Fully tested environment on AMD RX 7900
- Local inference with Safetensors (no CUDA)
- Shell scripts for easy setup, run & recovery
- Project-phase-structured folder layout

---

## 📁 Included

| File/Folder               | Description                                  |
|---------------------------|----------------------------------------------|
| `scripts/start_qwen_env.sh` | Activates venv, sets ROCm export vars         |
| `rocm_env/`               | Validated Python + Torch venv                |
| `README.md`               | This file :)                                |
| `.gitignore`              | Excludes large binaries & environment files |

---

## 🚀 Quickstart

```bash
git clone https://github.com/Ki-Horscht/rocm-6.1-qwen.git
cd rocm-6.1-qwen
bash scripts/start_qwen_env.sh
```

---

## 🧠 Use Case: Ethical Red Teaming & Analysis

- Derestriction Research (Bias, RLHF, Vocab)
- Controlled prompt response manipulation
- Reproducible token vector editing via Safetensors
- Critical prompts tested via `prompt_critical_tester_v6.py`

> 🛡️ All tools designed for ethical, controlled research.

---

## 📜 Setup Instructions

### 🌍 Ubuntu Preparation

```bash
sudo apt update
sudo apt install wget gnupg2 git nano micro
```

### Add ROCm 6.1 Repository (Stable)

```bash
wget https://repo.radeon.com/amdgpu-install/6.1.0/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb
sudo apt install ./amdgpu-install_6.1.60100-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
```

### Install ROCm Dev Packages

```bash
sudo apt install rocm-dev rocm-hip-runtime hip-runtime-amd rocm-smi
```

---

## 🌐 Environment Variables

```bash
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
```

---

## 📆 Python Environment Setup

```bash
python3.10 -m venv rocm_env
source rocm_env/bin/activate
pip install --upgrade pip
pip install "numpy<2.0.0"  # On 22.04
pip install torch==2.1.2+rocm6.1 torchvision==0.16.2+rocm6.1 --index-url https://download.pytorch.org/whl/rocm6.1
pip install transformers safetensors accelerate
```

On **Ubuntu 24.10**, newer numpy + transformers can be used.

---

## ✅ Testing

```bash
rocminfo | grep Name
/opt/rocm/bin/rocm-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:
```
GPU erkannt: True
Gerät: AMD Radeon RX 7900 XTX
```

---

## 🧪 Qwen Inference Test (7B)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-7B-Instruct")
inputs = tokenizer("Was ist die Hauptstadt von Peru?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚡ Known Issues & Fixes

| Problem                       | Fix                                              |
|------------------------------|--------------------------------------------------|
| ROCm not found               | Check if `/opt/rocm` → `/opt/rocm-6.1.1` symlink |
| Torch can't detect GPU       | Ensure all `EXPORT` vars are set                |
| numpy >=2.0 breaks legacy    | Downgrade: `pip install numpy<2.0`              |
| Kernel SDPA warnings         | Ignore for RX 7900 (Navi31), upstream WIP       |
| Safetensors import fails     | Use Python 3.10+ or latest pip fix              |

---

## 🔍 Project Phases

- **Phase 0**: System & ROCm Setup
- **Phase 1**: Shard Map + Head Detection
- **Phase 2**: Softfilter Removal (vector patcher)
- **Phase 3**: Deep Token Vector Patch (0.43-0.47 norms)
- **Phase 4**: Vocabulary Trigger & Obfuscation Scan
- **Phase 5**: RLHF / Reward Head Detection & Cleanup

> ⌛ During each phase, automated **Safetensor backups** via Bash (`smart_snapshot_exporter.py`) ensured rollback-safety.

---

## 🎯 Resources

- Required disk space: ~**24 GB** total (Model + Envs + Backup Scripts)
- Recommended system RAM: **32 GB+** for full inference on 7B / 32B
- Files were tested repeatedly using testprompts (prompt_testset_20.csv)

---

## 🤝 Contact

Maintainer: **KI-Horscht**  
GitHub: [https://github.com/KI-Horscht](https://github.com/KI-Horscht)  
Languages: English, Deutsch, العربية

---

## 🔒 Legal & Ethical Context

This repository is provided for **educational and research purposes** only.  
No derestriction code is published. Critical prompts were tested in controlled setups.

> ⚠️ ROCm inference on AMD is powerful – use it responsibly.

