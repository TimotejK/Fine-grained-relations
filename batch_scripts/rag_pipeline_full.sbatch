#!/bin/bash
#SBATCH --job-name=rag-pipeline-full
#SBATCH --time=8-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o out/rag_pipeline_full.out
#SBATCH -e out/rag_pipeline_full.err

# Create output directory if it doesn't exist
mkdir -p out

echo "=================================================="
echo "RAG Temporal Extraction Full Pipeline Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

cd "${HOME}/Fine-grained-relations"

# Function to check if previous step succeeded
check_exit_code() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1 failed with exit code $?"
        echo "Pipeline terminated at: $(date)"
        exit 1
    else
        echo "SUCCESS: $1 completed successfully"
    fi
}

# Stage 1: Generate Knowledge Base (if it doesn't exist)
echo ""
echo "=================================================="
echo "STAGE 1: Generating Knowledge Base"
echo "Start time: $(date)"
echo "=================================================="

if [ ! -f "rag_llm_approach/temporal_events_kb.json" ]; then
    echo "Knowledge base not found. Creating it..."
    srun \
      -A lpt -p frida \
      --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
      --container-mounts "$PWD":/workspace \
      bash -c "cd /workspace; python3 -c '
import json
import pandas as pd
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from datetime import datetime

# Load training data to create knowledge base
df = load_i2b2_absolute_data(test_split=False)
print(f\"Loaded {len(df)} training examples for KB creation\")

# Create knowledge base entries
kb_events = []
for idx, row in df.iterrows():
    try:
        event_text = row[\"text\"][row[\"start_char\"]:row[\"end_char\"]]

        # Convert minutes to readable time format
        base_date = datetime(1900, 1, 1, 0, 0, 0)
        start_time = base_date + pd.Timedelta(minutes=row[\"start_time_minutes\"])
        end_time = base_date + pd.Timedelta(minutes=row[\"end_time_minutes\"])

        kb_events.append({
            \"event_text\": event_text,
            \"event_type\": \"clinical_event\",
            \"temporal_expression\": f\"{start_time.strftime(\"%Y-%m-%d %H:%M\")} to {end_time.strftime(\"%Y-%m-%d %H:%M\")}\",
            \"normalized_time\": start_time.isoformat(),
            \"sentence_context\": row[\"text\"][:200] + \"...\" if len(row[\"text\"]) > 200 else row[\"text\"],
            \"document_id\": row[\"document_id\"]
        })

        if idx % 500 == 0:
            print(f\"Processed {idx}/{len(df)} events for KB\")
    except Exception as e:
        print(f\"Error processing event {idx}: {e}\")
        continue

# Save knowledge base
kb_data = {\"events\": kb_events}
with open(\"rag_llm_approach/temporal_events_kb.json\", \"w\") as f:
    json.dump(kb_data, f, indent=2)

print(f\"Knowledge base created with {len(kb_events)} events\")
print(\"KB saved to: rag_llm_approach/temporal_events_kb.json\")
'"

    check_exit_code "Knowledge Base Generation"
else
    echo "Knowledge base already exists, skipping generation"
fi

# Stage 2: Generate Fine-tuning Dataset
echo ""
echo "=================================================="
echo "STAGE 2: Generating Fine-tuning Dataset"
echo "Start time: $(date)"
echo "=================================================="

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; pip install google-genai ollama torchtext wandb; python3 rag_llm_approach/rag_absolute_time_extraction.py --mode generate_data"

check_exit_code "Fine-tuning Dataset Generation"

# Stage 3: Fine-tune the Model
echo ""
echo "=================================================="
echo "STAGE 3: Fine-tuning RAG Model"
echo "Start time: $(date)"
echo "=================================================="

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; python3 rag_llm_approach/rag_finetuning_pipeline.py --mode train --epochs 3 --batch_size 2 --output_dir ./rag_finetuned_model"

check_exit_code "Model Fine-tuning"

# Stage 4: Evaluate Base Model (now includes automatic evaluation)
echo ""
echo "=================================================="
echo "STAGE 4: Running Base Model Inference with Evaluation"
echo "Start time: $(date)"
echo "=================================================="

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; python3 rag_llm_approach/rag_absolute_time_extraction.py --mode inference"

check_exit_code "Base Model Inference and Evaluation"

# Stage 5: Evaluate Fine-tuned Model (now includes automatic evaluation)
echo ""
echo "=================================================="
echo "STAGE 5: Running Fine-tuned Model Inference with Evaluation"
echo "Start time: $(date)"
echo "=================================================="

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; python3 rag_llm_approach/rag_absolute_time_extraction.py --mode inference --use_finetuned"

check_exit_code "Fine-tuned Model Inference and Evaluation"

# Stage 6: Generate Results Summary
echo ""
echo "=================================================="
echo "STAGE 6: Generating Final Results Summary"
echo "Start time: $(date)"
echo "=================================================="

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/unsloth.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "cd /workspace; python3 -c '
import json
import os
from datetime import datetime

print(\"=\" * 50)
print(\"RAG TEMPORAL EXTRACTION PIPELINE RESULTS SUMMARY\")
print(\"=\" * 50)

# Check if all expected output files exist
files_to_check = [
    \"rag_llm_approach/temporal_events_kb.json\",
    \"rag_llm_approach/rag_finetuning_dataset.json\",
    \"rag_absolute_time_predictions.json\",
    \"rag_absolute_time_predictions_finetuned.json\"
]

print(\"\\nFile Generation Status:\")
for file_path in files_to_check:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f\"✓ {file_path} - {file_size:.2f} MB\")
    else:
        print(f\"✗ {file_path} - NOT FOUND\")

# Load and summarize results if available
try:
    if os.path.exists(\"rag_absolute_time_predictions.json\"):
        with open(\"rag_absolute_time_predictions.json\", \"r\") as f:
            base_results = json.load(f)
        print(f\"\\nBase Model Results: {len(base_results)} predictions\")

        # Count successful predictions
        successful = sum(1 for r in base_results if r.get(\"llm_response\", {}).get(\"start\") and r.get(\"llm_response\", {}).get(\"end\"))
        print(f\"Successful predictions: {successful}/{len(base_results)} ({successful/len(base_results)*100:.1f}%)\")

    if os.path.exists(\"rag_absolute_time_predictions_finetuned.json\"):
        with open(\"rag_absolute_time_predictions_finetuned.json\", \"r\") as f:
            finetuned_results = json.load(f)
        print(f\"\\nFine-tuned Model Results: {len(finetuned_results)} predictions\")

        # Count successful predictions
        successful_ft = sum(1 for r in finetuned_results if r.get(\"llm_response\", {}).get(\"start\") and r.get(\"llm_response\", {}).get(\"end\"))
        print(f\"Successful predictions: {successful_ft}/{len(finetuned_results)} ({successful_ft/len(finetuned_results)*100:.1f}%)\")

    if os.path.exists(\"rag_llm_approach/rag_finetuning_dataset.json\"):
        with open(\"rag_llm_approach/rag_finetuning_dataset.json\", \"r\") as f:
            dataset = json.load(f)
        print(f\"\\nFine-tuning Dataset: {len(dataset)} QA pairs generated\")

    if os.path.exists(\"rag_llm_approach/temporal_events_kb.json\"):
        with open(\"rag_llm_approach/temporal_events_kb.json\", \"r\") as f:
            kb = json.load(f)
        print(f\"Knowledge Base: {len(kb.get(\"events\", []))} events\")

except Exception as e:
    print(f\"Error reading results: {e}\")

print(f\"\\nPipeline completed at: {datetime.now()}\")
print(\"=\" * 50)
'"

check_exit_code "Results Summary Generation"

echo ""
echo "=================================================="
echo "RAG TEMPORAL EXTRACTION FULL PIPELINE COMPLETED"
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="
echo ""
echo "Output files generated:"
echo "- Knowledge Base: rag_llm_approach/temporal_events_kb.json"
echo "- Fine-tuning Dataset: rag_llm_approach/rag_finetuning_dataset.json"
echo "- Fine-tuned Model: ./rag_finetuned_model/"
echo "- Base Model Results: rag_absolute_time_predictions.json"
echo "- Fine-tuned Model Results: rag_absolute_time_predictions_finetuned.json"
echo "- Evaluation Results: rag_evaluation_results_*.json"
echo ""
echo "Check the output files for detailed results and performance metrics."
