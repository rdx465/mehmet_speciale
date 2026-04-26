#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap the NeuroVFM virtual environment on NGC CLD076.
#
# Run this ONCE on a worker node (worker01 or worker02) that has GPU access
# and /projects/ mounted.
#
# Usage:
#   ssh worker01
#   bash /path/to/templates/setup_venv_ngc.sh
#
# After setup, activate with:
#   source $VENV_PATH/bin/activate
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE="/projects/users/people/mehuns_r/projects/neurovfm_private"
VENV_PATH="${BASE}/venv"
NEXUS="https://nexus.mgmt.cld/repository/ngc-cloud-pypi/simple/"
LOG="${BASE}/logs/setup_venv_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${BASE}/logs"

echo "=== NeuroVFM venv setup ===" | tee -a "$LOG"
echo "Base : ${BASE}" | tee -a "$LOG"
echo "Venv : ${VENV_PATH}" | tee -a "$LOG"
echo "Log  : ${LOG}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── 1. Load modules ───────────────────────────────────────────────────────────
echo "── Step 1: Load modules" | tee -a "$LOG"
module purge
module load anaconda3/2024.02
module load cuda12.4/toolkit

python --version | tee -a "$LOG"
nvcc --version 2>/dev/null | grep "release" | tee -a "$LOG" || echo "nvcc not found" | tee -a "$LOG"

# ── 2. Create venv ────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Step 2: Create venv at ${VENV_PATH}" | tee -a "$LOG"
if [ -d "${VENV_PATH}" ]; then
    echo "  WARNING: venv already exists. Delete it first if you want a clean rebuild:" | tee -a "$LOG"
    echo "    rm -rf ${VENV_PATH}" | tee -a "$LOG"
    echo "  Continuing with existing venv..." | tee -a "$LOG"
else
    python -m venv "${VENV_PATH}"
    echo "  Created." | tee -a "$LOG"
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip --index-url "${NEXUS}" 2>&1 | tee -a "$LOG"

# ── 3. Discover available torch wheels on Nexus ───────────────────────────────
echo "" | tee -a "$LOG"
echo "── Step 3: Discover available torch versions on Nexus" | tee -a "$LOG"
python -m pip install "torch==" --index-url "${NEXUS}" 2>&1 | grep "from versions" | tee -a "$LOG" || true

# ── 4. Install PyTorch (cu124 — compatible with driver ≤ 12.7) ───────────────
echo "" | tee -a "$LOG"
echo "── Step 4: Install PyTorch 2.5.0+cu124" | tee -a "$LOG"
echo "  (cu124 requires CUDA driver >= 550.54, works with 12.7 drivers)" | tee -a "$LOG"

# Check driver first
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "  Driver CUDA version: ${DRIVER_CUDA}" | tee -a "$LOG"

python -m pip install \
    "torch==2.5.0+cu124" \
    "torchvision==0.20.0+cu124" \
    --index-url "${NEXUS}" \
    2>&1 | tee -a "$LOG"

# ── 5. Verify torch+CUDA before building torch-scatter ───────────────────────
echo "" | tee -a "$LOG"
echo "── Step 5: Verify torch+CUDA" | tee -a "$LOG"
python - <<'PYCHECK' 2>&1 | tee -a "$LOG"
import torch
print(f"  torch version  : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA runtime   : {torch.version.cuda}")
else:
    print("  WARNING: CUDA not available — torch-scatter will build CPU-only")
PYCHECK

# ── 6. Install torch-scatter (no build isolation so torch is visible) ─────────
echo "" | tee -a "$LOG"
echo "── Step 6: Install torch-scatter 2.1.2 (--no-build-isolation)" | tee -a "$LOG"
echo "  FORCE_CUDA=1 ensures CUDA kernels are compiled even if cuda.is_available()==False" | tee -a "$LOG"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0"   # H100; add "8.0;8.6" for A100/A10
export MAX_JOBS=4                    # parallel compile jobs

python -m pip install \
    "torch-scatter==2.1.2" \
    --no-build-isolation \
    --index-url "${NEXUS}" \
    2>&1 | tee -a "$LOG"

# ── 7. Install remaining NeuroVFM dependencies ────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Step 7: Install remaining dependencies" | tee -a "$LOG"

# outlines 1.1.1 first (has many sub-deps — install before neurovfm)
python -m pip install \
    "outlines==1.1.1" \
    --index-url "${NEXUS}" \
    2>&1 | tee -a "$LOG"

# neurovfm package (editable install from repo)
REPO_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
echo "  Installing neurovfm from: ${REPO_DIR}" | tee -a "$LOG"
python -m pip install -e "${REPO_DIR}" --index-url "${NEXUS}" 2>&1 | tee -a "$LOG"

# Remaining ML deps
python -m pip install \
    scikit-learn \
    scipy \
    matplotlib \
    tqdm \
    pandas \
    --index-url "${NEXUS}" \
    2>&1 | tee -a "$LOG"

# ── 8. Final verification ─────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Step 8: Final verification" | tee -a "$LOG"
python - <<'PYVERIFY' 2>&1 | tee -a "$LOG"
import importlib, sys

checks = [
    ("torch",         "torch.__version__"),
    ("torch_scatter", None),
    ("outlines",      "outlines.__version__"),
    ("sklearn",       "sklearn.__version__"),
    ("neurovfm",      None),
]

all_ok = True
for mod, ver_expr in checks:
    try:
        m = importlib.import_module(mod)
        ver = eval(ver_expr) if ver_expr else "ok"
        print(f"  OK  {mod:<20s} {ver}")
    except Exception as e:
        print(f"  FAIL {mod:<20s} {e}")
        all_ok = False

import torch
cuda_ok = torch.cuda.is_available()
print(f"\n  CUDA available : {cuda_ok}")
if cuda_ok:
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")

sys.exit(0 if all_ok else 1)
PYVERIFY

echo "" | tee -a "$LOG"
echo "=== Setup complete. Log saved to: ${LOG} ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "To activate the environment:" | tee -a "$LOG"
echo "  module load anaconda3/2024.02 cuda12.4/toolkit" | tee -a "$LOG"
echo "  source ${VENV_PATH}/bin/activate" | tee -a "$LOG"
