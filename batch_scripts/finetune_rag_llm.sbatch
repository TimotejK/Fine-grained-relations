#!/bin/bash
#SBATCH --job-name=fgtre-llm-finetune
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH -o out/llm_rag_finetune.out

cd "${HOME}/Fine-grained-relations"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; pip install google-genai ollama torchtext; python3 main.py --script finetune_rag_llm"
