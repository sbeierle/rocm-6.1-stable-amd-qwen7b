#!/bin/bash

echo "🔧 ROCm AutoFix gestartet…"

# === Virtuelle Umgebung aktivieren ===
if [ -d "$HOME/rocm_env" ]; then
    source "$HOME/rocm_env/bin/activate"
    echo "✅ Virtuelle Umgebung 'rocm_env' aktiviert."
else
    echo "❌ Kein 'rocm_env' gefunden unter $HOME/rocm_env"
    exit 1
fi

# === HIP & ROCm Variablen setzen ===
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

echo "✅ HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "✅ HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

# === Test ob GPU erkannt wird ===
python3 -c "import torch; print('torch:', torch.__version__); print('cuda.available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ Keine GPU erkannt')"

echo "🧠 ROCm AutoFix abgeschlossen."
