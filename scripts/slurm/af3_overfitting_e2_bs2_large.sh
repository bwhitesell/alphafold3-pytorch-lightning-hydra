#!/bin/bash -l

######################### Batch Headers #########################
#SBATCH --partition chengji-lab-gpu # use reserved partition `chengji-lab-gpu`
#SBATCH --account chengji-lab  # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres gpu:A100:1      # request A100 GPU resource(s)
#SBATCH --ntasks-per-node=1    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=59G              # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t 2-00:00:00          # time limit for the job (up to 28 days: `28-00:00:00`)
#SBATCH -J af3_overfitting_e2_bs2 # job name
#SBATCH --output=R-%x.%j.out   # output log file
#SBATCH --error=R-%x.%j.err    # error log file

module purge
module load cuda/11.8.0_gcc_9.5.0

# determine location of the project directory
use_private_project_dir=false # NOTE: customize as needed
if [ "$use_private_project_dir" = true ]; then
    project_dir="/home/$USER/data/Repositories/Lab_Repositories/alphafold3-pytorch-lightning-hydra"
else
    project_dir="/cluster/pixstor/chengji-lab/$USER/Repositories/Lab_Repositories/alphafold3-pytorch-lightning-hydra"
fi

# shellcheck source=/dev/null
source "/home/$USER/mambaforge/etc/profile.d/conda.sh"

cd "$project_dir" || exit
conda activate alphafold3-pytorch/

# Run training
srun python3 alphafold3_pytorch/train.py \
    data=pdb \
    data.batch_size=2 \
    data.overfitting_train_examples=true \
    data.sample_only_pdb_ids='[209d-assembly1, 721p-assembly1]' \
    experiment=alphafold3_overfitting_experiment \
    logger=wandb \
    +logger.wandb.entity=bml-lab \
    logger.wandb.group=alphafold3-overfitting-experiment \
    +logger.wandb.name=AlphaFold3-23M-Overfit-E2-BS2-Large-08222024 \
    model=alphafold3 \
    model.num_samples_per_example=5 \
    model.visualize_val_samples_every_n_steps=1 \
    +model.net.dim_pairwise=32 \
    +model.net.dim_single=96 \
    +model.net.dim_token=96 \
    +model.net.confidence_head_kwargs='{pairformer_depth: 1}' \
    +model.net.template_embedder_kwargs='{pairformer_stack_depth: 1}' \
    +model.net.msa_module_kwargs='{depth: 1, dim_msa: 4}' \
    +model.net.pairformer_stack='{depth: 2, pair_bias_attn_dim_head: 32, pair_bias_attn_heads: 8}' \
    +model.net.diffusion_module_kwargs='{atom_encoder_depth: 2, atom_encoder_heads: 4, token_transformer_depth: 8, token_transformer_heads: 16, atom_decoder_depth: 2, atom_decoder_heads: 4, atom_encoder_kwargs: {attn_pair_bias_kwargs: {dim_head: 16}}, atom_decoder_kwargs: {attn_pair_bias_kwargs: {dim_head: 16}}}' \
    trainer.check_val_every_n_epoch=null \
    +trainer.val_check_interval=50 \
    +trainer.log_every_n_steps=1

# Inform user of run completion
echo 'Run completed.'
