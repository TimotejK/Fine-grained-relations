# Fine-Grained Temporal Information Extraction from Medical Documents

This project explores various approaches for extracting fine-grained temporal information from medical documents, with a focus on identifying event timelines, durations, and absolute temporal anchoring. The system experiments with traditional machine learning models, transformer-based architectures, and LLM-based approaches with and without retrieval-augmented generation (RAG).

## 🎯 Project Goal

Extract precise temporal information from medical text, including:
- **Start times** of medical events (e.g., when a treatment began)
- **End times** of medical events (e.g., when symptoms resolved)
- **Durations** (e.g., how long a condition lasted)
- **Temporal expressions** and their normalization
- **Event timelines** from patient records

## 📁 Project Structure

```
Fine-grained-relations/
├── data_loaders/          # Data loading and preprocessing
│   ├── load_i2b2_data_updated.py
│   ├── thyme_loader.py
│   ├── preprocessing.py
│   └── dataset.py
├── models/                # Traditional ML and transformer models
│   ├── BertBasedModel.py
│   ├── SimplifiedBertBasedModel.py
│   ├── LSTMBasedModel.py
│   └── ClosestBertBasedModel.py
├── rag_llm_approach/      # RAG-based LLM extraction
│   ├── rag_absolute_time_extraction.py
│   ├── build_knowledge_graph.py
│   ├── extract_event_timelines.py
│   ├── evaluate_all_results.py
│   └── analyze_retrieval_stats.py
├── llm_finetuning/        # LLM fine-tuning experiments
│   ├── finetuning.py
│   ├── hyperparameter_tuning.py
│   └── model_testing.py
├── evaluation/            # Evaluation metrics and scripts
│   ├── metrics.py
│   ├── evaluate.py
│   └── error_analysis.py
├── batch_scripts/         # SLURM batch scripts for HPC
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for LLM experiments)
- Access to i2b2 or THYME medical datasets

### Installation

```bash
# Clone the repository
git clone https://github.com/UL-FRI-Knez/Fine-Grained-Temporal-Information-Extraction
cd Fine-Grained-Temporal-Information-Extraction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Set up the following environment variables:

```bash
export WANDB_API_KEY="your_wandb_key"  # For experiment tracking
export OPENAI_API_KEY="your_openai_key"  # For GPT-based experiments
```

## 🔬 Experimental Approaches

### 1. Traditional Transformer Models (`models/`)

Classical supervised learning approaches using BERT and LSTM architectures:

- **BertBasedModel**: Multi-task BERT model predicting temporal units and values
- **SimplifiedBertBasedModel**: Streamlined version focusing on core predictions
- **LSTMBasedModel**: LSTM-based sequence model for temporal extraction
- **ClosestBertBasedModel**: Uses closest temporal expression as reference

**Training:**
```bash
python main.py --script train_bert_model
python main.py --script train_lstm_model
```

### 2. LLM Fine-tuning (`llm_finetuning/`)

Fine-tuning large language models (Llama, Gemma) for temporal extraction:

**Features:**
- Parameter-efficient fine-tuning with LoRA/QLoRA
- Hyperparameter optimization with Optuna
- Support for various LLM architectures
- Weights & Biases integration

**Quick Start:**
```bash
# Hyperparameter tuning
python -m llm_finetuning.hyperparameter_tuning

# Fine-tune with best parameters
python main.py --script finetune_llm --model unsloth/Meta-Llama-3.1-8B

# Test fine-tuned model
python -m llm_finetuning.model_testing
```

See [`llm_finetuning/QUICKSTART.md`](llm_finetuning/QUICKSTART.md) for detailed instructions.

### 3. RAG-Enhanced LLM Approach (`rag_llm_approach/`)

Retrieval-Augmented Generation using a knowledge base of temporal events to improve extraction accuracy.

Uses extracted patient timelines with surrounding events for richer context:
```bash
python rag_llm_approach/rag_absolute_time_extraction.py \
    --mode generate_data \
    --use_timelines \
    --timeline_file rag_llm_approach/patient_timelines.json
```

**Features:**
- Retrieves similar events from knowledge base/timelines
- Provides context including absolute times and temporal expressions
- Marks target events with XML tags for clarity
- Generates fine-tuning datasets with RAG-enhanced prompts

**Workflow:**
```bash
# 1. Build knowledge base
python rag_llm_approach/build_knowledge_graph.py

# 2. Extract patient timelines (optional, for timeline mode)
python rag_llm_approach/extract_event_timelines.py

# 3. Generate RAG-enhanced training data
python rag_llm_approach/rag_absolute_time_extraction.py --mode generate_data

# 4. Fine-tune LLM with RAG data
python main.py --script finetune_rag_llm

# 5. Evaluate results
python rag_llm_approach/evaluate_all_results.py
```

See [`rag_llm_approach/README_TIMELINE_MODE.md`](rag_llm_approach/README_TIMELINE_MODE.md) for more details.

## 📊 Evaluation

### Metrics

The project uses several evaluation metrics:
- **Accuracy**: Proportion of correctly predicted temporal values
- **Mean Absolute Error (MAE)**: Average absolute difference from ground truth
- **F1 Score**: For temporal unit classification
- **95% Confidence Intervals**: Computed via bootstrapping (1000 samples)

### Running Evaluation

```bash
# Evaluate all RAG experiments
python rag_llm_approach/evaluate_all_results.py

# Analyze retrieval statistics
python rag_llm_approach/analyze_retrieval_stats.py

# Error analysis
python evaluation/error_analysis.py
```

Results are saved in the `results/` directory and formatted as LaTeX tables for publication.

## 🗄️ Data

The project supports two medical temporal datasets:

- **i2b2 2012 Temporal Relations Challenge**: Clinical narratives with temporal annotations
- **THYME Corpus**: Cancer-focused clinical notes with rich temporal information

Data loaders automatically handle:
- XML parsing and sanitization
- Temporal expression extraction (TIMEX3)
- Event annotation processing
- Train/test splitting

## 📈 Results & Analysis

Results are tracked using:
- **Weights & Biases**: Real-time experiment tracking
- **Local logs**: Detailed evaluation outputs
- **Excel/LaTeX tables**: Publication-ready result summaries

Analysis notebooks available in `Analiza/` and `notebooks/`.

## 🔧 Configuration

Key configuration files:
- `models/model_config.py`: Model hyperparameters
- `requirements.txt`: Python dependencies
- `batch_scripts/*.sbatch`: HPC job configurations
