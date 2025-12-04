#!/bin/bash
#SBATCH --job-name=fgtre-bert
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o out/bert_train.out

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/modern_bert.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; pip install dspy trl google-genai ollama torchtext spacy 'scipy<1.11'; python3 -m spacy download en_core_web_sm; python3 main.py --script train_bert_model"
