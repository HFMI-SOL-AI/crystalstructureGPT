#!/bin/bash
#SBATCH --account=solai
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=booster
#SBATCH --output=/p/project1/solai/chandran1/structureGPT/train-%j.out
#SBATCH --error=/p/project1/solai/chandran1/structureGPT/train-%j.err

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR}i"

cd /p/project1/solai/chandran1/structureGPT/
source /p/project1/solai/chandran1/structureGPT/.venv/bin/activate
# python src/nanogpt/train.py --eval_interval=30 --dataset=cst/out --batch_size=256 --block_size=128 --compile=False
torchrun --standalone --nproc_per_node=4 src/nanogpt/train.py --eval_interval=20 --dataset=cst/out --batch_size=128 --block_size=200 --compile=False
