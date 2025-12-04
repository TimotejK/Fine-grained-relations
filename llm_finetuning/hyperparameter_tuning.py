import re
from datetime import datetime
import os


import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
from datasets import Dataset
from unsloth import FastLanguageModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import compute_metrics
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig
import gc

folder = "./data/"
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
EOS_TOKEN = None


def log_text(line, log_file="hyperparameter_tuning.log"):
    with open(log_file, "a") as log:
        log.write(line)
        log.write("\n")


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def prepare_model(model_name, max_seq_length, r, lora_alpha, lora_dropout, use_cpu_offload=True):
    global EOS_TOKEN
    dtype = None
    load_in_4bit = True

    # Prepare kwargs for model loading with proper quantization configuration
    load_kwargs = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
    }

    # For quantized models, we need to be careful with device mapping
    # The unsloth library handles device mapping internally, so we don't override it
    # Instead, we'll rely on the default behavior which is optimized for the library

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    if "gemma" in model_name:
        model = FastLanguageModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=3407,
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    EOS_TOKEN = tokenizer.eos_token

    return model, tokenizer


def prepare_data(model_name, tokenizer, use_absolute_qa=True, validation_split=0.1):
    """Prepare both training and validation datasets"""
    df = load_i2b2_absolute_data(test_split=False)

    # Split into train and validation
    total_size = len(df)
    val_size = int(total_size * validation_split)
    train_df = df[val_size:]
    val_df = df[:val_size]

    # Prepare training dataset
    train_datasetPT = TimelineDataset(train_df, use_qa_format=True, use_absolute_qa=use_absolute_qa)

    def dataset_generator(ptDataset):
        for i in range(len(ptDataset)):
            yield ptDataset[i]

    train_dataset = Dataset.from_generator(lambda: dataset_generator(train_datasetPT))

    # Prepare validation dataset
    val_datasetPT = TimelineDataset(val_df, use_qa_format=True, use_absolute_qa=use_absolute_qa)
    val_dataset = Dataset.from_generator(lambda: dataset_generator(val_datasetPT))

    if "gemma" in model_name:
        def apply_chat_template_to_example(example):
            # Convert to conversation format
            conversations = [
                {"content": example["instruction"] + "\n\n\nInput:\n" + example["input"], "role": "user"},
                {"content": example["output"], "role": "assistant"},
            ]
            # Apply chat template
            text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False
            )
            # Keep original fields and add text
            example["text"] = text
            example["conversations"] = conversations
            return {"text": text}

        train_dataset = train_dataset.map(apply_chat_template_to_example)
        val_dataset = val_dataset.map(apply_chat_template_to_example)
    else:
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    return train_dataset, val_dataset


def parse_answer_absolute_time(answer):
    start_match = re.search(r"START:\s*([^\.,\n\<]+)", answer, re.IGNORECASE)
    end_match = re.search(r"END:\s*([^\.,\n\<]+)", answer, re.IGNORECASE)

    def to_minutes_iso(time_str):
        try:
            dt = datetime.fromisoformat(time_str.strip())
            base = datetime(1900, 1, 1, 0, 0, 0)
            delta = dt - base
            return int(delta.total_seconds() // 60)
        except Exception:
            return 0

    start = to_minutes_iso(start_match.group(1)) if start_match else 0
    end = to_minutes_iso(end_match.group(1)) if end_match else 0
    return {"start": start, "end": end}


def extract_model_answer(text):
    match = re.findall(r"<start_of_turn>model\n(.*?)<end_of_turn>", text, re.DOTALL)
    if match:
        return match[-1].strip()
    return text


def evaluate_model_on_validation(model, val_dataset, tokenizer, model_name, use_absolute_qa=True, compute_loss=False):
    """Evaluate the model on validation set and return metrics"""
    FastLanguageModel.for_inference(model)

    predictions_starts = []
    predictions_ends = []
    predictions_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []
    total_loss = 0.0
    loss_count = 0

    # Sample a subset for faster evaluation during hyperparameter tuning
    sample_size = min(10, len(val_dataset))  # Evaluate on 10 examples for better approximation

    for i, example in enumerate(val_dataset):
        if i >= sample_size:
            break

        # Clear cache every 3 examples to prevent accumulation
        if i > 0 and i % 3 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if "gemma" in model_name:
            # For gemma, we need to reconstruct the prompt without the answer
            conversations = example.get("conversations", [])
            if len(conversations) > 1:
                prompt_conversations = conversations[:-1]
                formatted_text = tokenizer.apply_chat_template(
                    prompt_conversations,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                formatted_text = example["text"]
        else:
            formatted_text = alpaca_prompt.format(
                example["instruction"],
                example["input"],
                "",
            )

        inputs = tokenizer([formatted_text], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        answer = tokenizer.batch_decode(outputs)[0]

        # Clean up intermediate tensors immediately
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if "gemma" in model_name:
            answer = extract_model_answer(answer)

        parsed = parse_answer_absolute_time(answer)
        s = parsed["start"]
        predictions_starts.append(s)
        e = parsed["end"]
        predictions_ends.append(e)
        d = parsed["end"] - parsed["start"]
        predictions_durations.append(d)

        gs = (example["row"]["start_time_minutes"],
              example["row"]["start_lower_minutes"],
              example["row"]["start_upper_minutes"])
        gold_starts.append(gs)
        ge = (example["row"]["end_time_minutes"],
              example["row"]["end_lower_minutes"],
              example["row"]["end_upper_minutes"])
        gold_ends.append(ge)
        gd = (example["row"]["duration_minutes"],
              example["row"]["duration_lower_minutes"],
              example["row"]["duration_upper_minutes"])
        gold_durations.append(gd)

    eval_metrics = compute_metrics(
        predictions_starts, predictions_ends, predictions_durations,
        gold_starts, gold_ends, gold_durations
    )

    # Add validation loss if computed
    if compute_loss and loss_count > 0:
        eval_metrics["validation_loss"] = total_loss / loss_count

    return eval_metrics


def objective(trial, model_name, max_seq_length, use_absolute_qa=True):
    """Optuna objective function for hyperparameter tuning"""

    # Monitor GPU memory at start of trial
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nTrial {trial.number} starting - GPU Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        log_text(f"Trial {trial.number} starting - GPU Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB",
                 log_file="hyperparameter_tuning.log")

    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [2, 4])  # Larger batch for faster training
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [4, 8])  # Reduced for faster training
    warmup_steps = trial.suggest_int("warmup_steps", 5, 15)  # Reduced range
    max_steps = trial.suggest_int("max_steps", 50, 200)  # Use max_steps instead of epochs for better control
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # LoRA parameters
    r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)

    # Initialize wandb for this trial
    run = wandb.init(
        project="fine-grained-hyperparameter-tuning",
        config={
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "weight_decay": weight_decay,
            "lora_r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        reinit=True,
    )

    # Initialize variables to avoid UnboundLocalError in finally block
    model = None
    tokenizer = None
    trainer = None
    train_dataset = None
    val_dataset = None
    objective_value = float('inf')  # Default value if error occurs

    try:
        # Force cleanup before loading model to ensure clean state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Prepare model with trial hyperparameters
        model, tokenizer = prepare_model(model_name, max_seq_length, r, lora_alpha, lora_dropout)

        # Prepare datasets
        train_dataset, val_dataset = prepare_data(model_name, tokenizer, use_absolute_qa)

        # Training arguments with trial hyperparameters
        if "gemma" in model_name:
            training_args = SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,  # Use max_steps instead of epochs for faster trials
                learning_rate=learning_rate,
                logging_steps=25,  # Log less frequently
                optim="adamw_8bit",
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=f"outputs/trial_{trial.number}",
                report_to="wandb",  # Enable wandb logging for training metrics
                gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
                max_grad_norm=1.0,
                save_strategy="no",  # Don't save checkpoints during hyperparameter tuning
                logging_first_step=True,
            )
        else:
            training_args = TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,  # Use max_steps instead of epochs for faster trials
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=25,  # Log less frequently
                optim="adamw_8bit",
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=f"outputs/trial_{trial.number}",
                report_to="wandb",  # Enable wandb logging for training metrics
                gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
                max_grad_norm=1.0,
                save_strategy="no",  # Don't save checkpoints during hyperparameter tuning
                logging_first_step=True,
            )

        # Create trainer
        trainer = SFTTrainer(
            dataset_text_field="text",
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )

        if "gemma" in model_name:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )

        # Train the model
        train_result = trainer.train()

        # Log final training loss
        if hasattr(train_result, 'training_loss'):
            final_training_loss = train_result.training_loss
            wandb.log({"final_training_loss": final_training_loss})
            log_text(f"Trial {trial.number} final training loss: {final_training_loss}", log_file="hyperparameter_tuning.log")

        # Evaluate on validation set
        eval_metrics = evaluate_model_on_validation(model, val_dataset, tokenizer, model_name, use_absolute_qa, compute_loss=True)

        # Immediately clear CUDA cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log metrics to wandb
        wandb_metrics = {
            "val_start_mae": eval_metrics["start_mae"],
            "val_end_mae": eval_metrics["end_mae"],
            "val_duration_mae": eval_metrics["duration_mae"],
            "val_start_accuracy": eval_metrics["start_accuracy"],
            "val_end_accuracy": eval_metrics["end_accuracy"],
            "val_duration_accuracy": eval_metrics["duration_accuracy"],
        }

        # Add validation loss if available
        if "validation_loss" in eval_metrics:
            wandb_metrics["validation_loss"] = eval_metrics["validation_loss"]

        wandb.log(wandb_metrics)

        # Log results
        log_text(f"Trial {trial.number}: {trial.params}", log_file="hyperparameter_tuning.log")
        log_text(f"Trial {trial.number} metrics: {eval_metrics}", log_file="hyperparameter_tuning.log")

        # Use a combination of metrics as the objective
        # Lower MAE and higher accuracy are better
        # We'll use a weighted combination (you can adjust weights)
        objective_value = (
            eval_metrics["start_mae"] +
            eval_metrics["end_mae"] +
            eval_metrics["duration_mae"] -
            1000 * (eval_metrics["start_accuracy"] +
                    eval_metrics["end_accuracy"] +
                    eval_metrics["duration_accuracy"])
        ) / 6.0

        wandb.log({"objective_value": objective_value})

    except Exception as e:
        log_text(f"Trial {trial.number} failed with error: {str(e)}", log_file="hyperparameter_tuning.log")
        print(f"Trial {trial.number} failed with error: {str(e)}")
        objective_value = float('inf')

    finally:
        # Aggressive memory cleanup to prevent OOM on subsequent trials
        print(f"Cleaning up memory for trial {trial.number}...")

        # Clean up trainer first (it holds references to model and optimizer)
        if trainer is not None:
            try:
                # Clear optimizer state
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    del trainer.optimizer
                if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
                    del trainer.lr_scheduler
                # Clear trainer state
                if hasattr(trainer, 'state'):
                    del trainer.state
            except:
                pass
            del trainer
            trainer = None

        # Clean up model
        if model is not None:
            try:
                # Move model to CPU first to free GPU memory
                model.cpu()
                # Clear model parameters
                if hasattr(model, 'base_model'):
                    del model.base_model
            except:
                pass
            del model
            model = None

        # Clean up datasets
        if train_dataset is not None:
            del train_dataset
            train_dataset = None
        if val_dataset is not None:
            del val_dataset
            val_dataset = None

        # Clean up tokenizer
        if tokenizer is not None:
            del tokenizer
            tokenizer = None

        # Force CUDA cleanup
        if torch.cuda.is_available():
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            # Empty cache multiple times (sometimes needed)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            # Reset peak memory stats for monitoring
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        # Force garbage collection multiple times
        gc.collect()
        gc.collect()

        # Finish wandb run
        try:
            wandb.finish()
        except:
            pass

        # Print memory stats for debugging
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory after cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Add a small delay to ensure memory is fully released
        import time
        time.sleep(2)

    return objective_value


def run_hyperparameter_tuning(model_name, n_trials=20, max_seq_length=4096, use_absolute_qa=True):
    """Run Optuna hyperparameter tuning"""

    # Create study with versioned name to avoid conflicts with old hyperparameter ranges
    # Generate a timestamp or version identifier
    study_version = "v3"  # Increment this if you change hyperparameter ranges again

    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"llm-finetuning-hyperparameter-optimization-{study_version}",
        storage=f"sqlite:///llm_finetuning/optuna_study_{model_name.split('/')[-1]}_{study_version}.db",
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, model_name, max_seq_length, use_absolute_qa),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print best results
    print("\n" + "="*80)
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (objective): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Log best parameters
    log_text("\n" + "="*80, log_file="hyperparameter_tuning.log")
    log_text("Best trial:", log_file="hyperparameter_tuning.log")
    log_text(f"  Value (objective): {trial.value}", log_file="hyperparameter_tuning.log")
    log_text("  Params:", log_file="hyperparameter_tuning.log")
    for key, value in trial.params.items():
        log_text(f"    {key}: {value}", log_file="hyperparameter_tuning.log")

    # Save best parameters to a file
    import json
    model_short_name = model_name.split('/')[-1]
    best_params_path = f"llm_finetuning/best_hyperparameters_{model_short_name}_{study_version}.json"
    with open(best_params_path, "w") as f:
        json.dump(trial.params, f, indent=2)

    print(f"\nBest hyperparameters saved to: {best_params_path}")
    log_text(f"\nBest hyperparameters saved to: {best_params_path}", log_file="hyperparameter_tuning.log")

    # Plot optimization history (if plotly is available)
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"llm_finetuning/optimization_history_{model_short_name}_{study_version}.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"llm_finetuning/param_importances_{model_short_name}_{study_version}.html")

        print(f"Optimization plots saved as HTML files")
    except ImportError:
        print("Plotly not available for visualization. Install with: pip install plotly")

    return study

def main():
    # Example usage
    # model_name = "unsloth/Meta-Llama-3.1-8B"
    model_name = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"

    study = run_hyperparameter_tuning(
        model_name=model_name,
        n_trials=20,  # Number of trials to run
        max_seq_length=2048,
        use_absolute_qa=True
    )

    print("\nHyperparameter tuning completed!")

if __name__ == '__main__':
    main()

