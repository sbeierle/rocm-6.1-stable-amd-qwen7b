#!/bin/bash
# Start Qwen ROCm environment
source ~/rocm_env/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
cd ~/models/qwen-32B-Instruct
echo "✅ Ready for inference"
