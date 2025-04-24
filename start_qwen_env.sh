# Datei: ~/source_qwen_env.sh
# ✨ Dies ist ein Source-Skript – NICHT ausführbar machen, sondern nur sourcen!

echo "🌀 Aktiviere venv & setze ROCm Variablen …"

# venv aktivieren
source ~/rocm_env/bin/activate

# ROCm Variablen
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Wechsle ins Projektverzeichnis
cd ~/qwen-project || { echo "❌ qwen-project nicht gefunden!"; return 1; }

echo "✅ Umgebung ist bereit in: $(pwd)"

# GPU Check
echo -e "\n🔍 GPU Check:"
python -c "import torch; print('GPU erkannt:', torch.cuda.is_available()); print('Gerät:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
