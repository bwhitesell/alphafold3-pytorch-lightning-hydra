#!/bin/bash

######################### Batch Headers #########################
#SBATCH --partition=gpu                                       # use partition `gpu` for GPU nodes
#SBATCH --account=pawsey1018-gpu                              # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=1                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres=gpu:1                                          # NOTE: requests any GPU resource(s)
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time 0-24:00:00                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=af3_overfitting_e1_bs1_large               # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#SBATCH --signal=SIGUSR1@90                                   # send SIGUSR1 90 seconds before job end to trigger job resubmission
# NOTE: One cannot request exclusive access to a node
#################################################################

# Load required modules
module load pawseyenv/2023.08
module load singularity/3.11.4-slurm

# Determine directory paths
export MIOPEN_USER_DB_PATH="/scratch/director2187/amorehead/tmp/my-miopen-cache/af3_rocm"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Create user database path
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
export containerImage=/scratch/pawsey1018/amorehead/af3-pytorch-lightning-hydra/af3-pytorch-lightning-hydra_0.3.8_dev.sif

# Set up WandB run
RUN_ID="fidch6u7"  # NOTE: Generate a unique ID for each run using `python3 scripts/generate_id.py`

# Run container
srun singularity exec --rocm \
    --cleanenv \
    -H "$PWD":/home \
    -B alphafold3-pytorch-lightning-hydra:/alphafold3-pytorch-lightning-hydra \
    --pwd /alphafold3-pytorch-lightning-hydra \
    "$containerImage" \
    WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python3 alphafold3_pytorch/train.py experiment=af3_overfitting_e1_bs1_large

# Inform user of run completion
echo "Run completed for job: $SLURM_JOB_NAME."
