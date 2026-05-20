#!/bin/bash
# PBS GPU Job Template -- NeuroVFM pipeline on NGC CLD076
#
# Usage:
#   qsub -W group_list=icp_users -A icp_users templates/GPUJob.sh
#   qsub -W group_list=icp_users -A icp_users -v SINGLE=1-154 templates/GPUJob.sh
#   qsub -W group_list=icp_users -A icp_users -v SCRIPT=train_mil_head_ngc.py templates/GPUJob.sh

# Resource requests
#PBS -N neurovfm_pipeline
#PBS -l nodes=1:ppn=4:gpu
#PBS -l walltime=12:00:00
#PBS -l mem=32gb
#PBS -j oe
#PBS -o /projects/users/people/mehuns_r/projects/neurovfm_private/logs/job_output.log

# Paths
BASE="/projects/users/people/mehuns_r/projects/neurovfm_private"
REPO="/projects/users/people/mehuns_r/projects/neurovfm_private/mehmet_speciale"
SCRIPTS="${REPO}/scripts"

# Environment
module purge
module load anaconda3/2024.02
module load cuda12.4/toolkit

conda activate neurovfm_gpu

# Verify GPU
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PYCHECK'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYCHECK

cd "${REPO}"

# Select script
SCRIPT="${SCRIPT:-extract_features_ngc.py}"

ARGS=""
if [ -n "${SINGLE}" ]; then
    ARGS="--single ${SINGLE}"
fi

echo "Running: python ${SCRIPTS}/${SCRIPT} ${ARGS}"
echo "Job ID : ${PBS_JOBID}"
echo "Node   : $(hostname)"
echo "Started: $(date)"

python "${SCRIPTS}/${SCRIPT}" ${ARGS}

EXIT_CODE=$?
echo "Finished: $(date)  |  Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
