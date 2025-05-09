#!/bin/bash
#SBATCH --job-name=fgtre-bert
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L4:1
#SBATCH -o bert_train.out

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/modern_bert.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; python3 main.py --script train_bert_model"
