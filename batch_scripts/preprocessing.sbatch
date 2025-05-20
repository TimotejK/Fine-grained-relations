#!/bin/bash
#SBATCH --job-name=preprocessing-i2b2
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -o preprocessing.out

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/modern_bert.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; apt update; apt install -y curl pciutils; curl -fsSL https://ollama.com/install.sh | sh; pip install dspy trl; python3 main.py --script preprocess"
