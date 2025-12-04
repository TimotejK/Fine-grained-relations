#!/bin/bash
#SBATCH --job-name=hyperparameter-optimization
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH -o out/optuna.out
#SBATCH -e out/optuna.err

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; pip install dspy trl google-genai ollama torchtext optuna optuna-integration[wandb]; python3 main.py --script finetuning_hyperparameter_search"
