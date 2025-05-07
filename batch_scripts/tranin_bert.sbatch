#!/bin/bash
#SBATCH --job-name=fgtre-bert
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o bert_train.out

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/gold.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "pip install wandb; python main.py --script train_bert_model"
