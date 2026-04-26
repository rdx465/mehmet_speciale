#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# PBS GPU Job Template — NeuroVFM pipeline on NGC CLD076
#
# Usage:
#   qsub templates/GPUJob.sh                          # feature extraction (all)
#   qsub -v SINGLE=1-154 templates/GPUJob.sh          # single patient (testing)
#   qsub -v SCRIPT=train_mil_head_ngc.py templates/GPUJob.sh  # downstream ML
#
# GPU nodes: run  `pbsnodes -a | grep -A5 gpu`  on login node to list available
# ─────────────────────────────────────────────────────────────────────────────

# ── Resource requests ─────────────────────────────────────────────────────────
#PBS -N neurovfm_pipeline
#PBS -l nodes=1:ppn=4:gpu
#PBS -l walltime=12:00:00
#PBS -l mem=32gb
#PBS -j oe
#PBS -o /projects/users/people/mehuns_r/projects/neurovfm_private/logs/job_${PBS_JOBID}.log

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE="/projects/users/people/mehuns_r/projects/neurovfm_private"
REPO="/projects/users/people/mehuns_r/projects/neurovfm_lokal/mehmet_speciale"
VENV_PATH="${BASE}/venv"
SCRIPTS="${REPO}/scripts"

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02
module load cuda12.4/toolkit

source "${VENV_PATH}/bin/activate"

# Verify GPU is visible
echo "── GPU check ──────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PYCHECK'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
PYCHECK
echo "───────────────────────────────────────────────────────────────────────"

cd "${REPO}"

# ── Select script and arguments ───────────────────────────────────────────────
# Default: feature extraction for all patients
SCRIPT="${SCRIPT:-extract_features_ngc.py}"

ARGS=""
if [ -n "${SINGLE}" ]; then
    ARGS="--single ${SINGLE}"
fi

echo "Running: python ${SCRIPTS}/${SCRIPT} ${ARGS}"
echo "Job ID : ${PBS_JOBID}"
echo "Node   : $(hostname)"
echo "Started: $(date)"
echo "───────────────────────────────────────────────────────────────────────"

python "${SCRIPTS}/${SCRIPT}" ${ARGS}

EXIT_CODE=$?
echo "───────────────────────────────────────────────────────────────────────"
echo "Finished: $(date)  |  Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
