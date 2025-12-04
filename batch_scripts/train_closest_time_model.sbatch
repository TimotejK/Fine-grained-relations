#!/bin/bash
#SBATCH --job-name=time-finder-bert
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o out/time_finder_train.out

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/modern_bert.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; python3 main.py --script train_time_finder"
