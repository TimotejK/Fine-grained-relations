import json
import os
from pathlib import Path

import torch
import datetime
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.evaluate import evaluate
from models import BertBasedModel, SimplifiedBertBasedModel, LSTMBasedModel, ClosestBertBasedModel
from models.model_config import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps") if torch.backends.mps.is_available() else device

def train(model, dataset, dataset_test, config, project_name="timeline_training"):
    import wandb
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model.to(device)

    simplified_model = config.model_type == "simplified_transformer" or config.model_type == "lstm" or config.model_type == "closest_transformer"
    epochs = config.training_hyperparameters["epochs"]
    batch_size = config.training_hyperparameters["batch_size"]
    lr = config.training_hyperparameters["learning_rate"]
    weight_decay = config.training_hyperparameters["weight_decay"]
    step_size = config.training_hyperparameters["scheduler_config"]["step_size"]
    gamma = config.training_hyperparameters["scheduler_config"]["gamma"]

    # Set random seed for reproducibility
    seed = config.training_hyperparameters["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    wandb.init(project=project_name, config={
        "model_name": model.encoder.__class__.__name__,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    })
    wandb.run.name = run_name

    wandb.watch(model, log="all", log_freq=10)
    wandb.config.update(config.to_dict())  # Log the model configuration to wandb

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    total_steps = len(dataloader) * epochs // batch_size
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # Setup gradient clipping
    max_grad_norm = 1.0

    # Setup early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    # Training loop
    model.train()

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            if simplified_model:
                labels = (
                    batch['start_time_minutes'] - batch['admission_date_minutes'],
                    batch['end_time_minutes'] - batch['admission_date_minutes'],
                    batch['duration_minutes']
                )
            else:
                labels = (
                    batch['start_unit'].to(device),
                    batch['start_value'].to(device),
                    batch['end_unit'].to(device),
                    batch['end_value'].to(device),
                    batch['duration_unit'].to(device),
                    batch['duration_value'].to(device)
                )

            # Forward pass
            try:
                if config.model_type == "simplified_transformer":
                    outputs = model(
                        text=batch['text'],
                        labels=labels,
                        start_char_index=batch['start_char'],
                        end_char_index=batch['end_char']
                    )
                    loss = outputs['loss']
                elif config.model_type == "closest_transformer":
                    outputs = model(
                        text=batch['text'],
                        labels=labels,
                        start_char_index=batch['start_char'],
                        end_char_index=batch['end_char'],
                        temporal_expressions = batch['temporal_expressions'],
                        admission_date_minutes = batch['admission_date_minutes']
                    )
                    loss = outputs['loss']

                # Check for NaN loss
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()

                # Track loss
                total_loss += loss.item()

            except Exception as e:
                print(f"Error in batch: {e}")
                continue

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(dataloader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        eval_loss, eval_metrics = evaluate(model, dataloader_test, simplified_model=simplified_model, config=config)

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "eval_loss": eval_loss,
            "eval_metrics": eval_metrics,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {eval_loss:.4f}")
        print(f"  Metrics: {eval_metrics}")

        # Log to file
        with open("runs.txt", "a") as log_file:
            log_entry = f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {eval_loss:.4f}, Metrics: {eval_metrics}\n"
            log_file.write(log_entry)

        # Early stopping check
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            patience_counter = 0
            # Save best model
            save_model(model, model.tokenizer, f"results/{run_name}_best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save final model
    save_model(model, model.tokenizer, f"results/{run_name}_final")
    evaluate(model, dataloader_test, simplified_model=simplified_model, save_error_analysis=True, model_id=run_name, config=config)
    wandb.finish()
    return eval_metrics

def save_model(model, tokenizer, target):
    Path(target).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{target}/model_state_dict.pth")
    tokenizer.save_pretrained(target + "/tokenizer")
    with open(f"{target}/model_config.json", "w") as f:
        json.dump(model.config.to_dict(), f)
    with open(f"{target}/training_config.json", "w") as f:
        json.dump(model.model_config.to_dict(), f)

def load_model(directory, device=torch.device('cpu')):
    with open(f"{directory}/model_config.json", "r") as f:
        config = json.load(f)
    with open(f"{directory}/training_config.json", "r") as f:
        model_config = ModelConfig.from_dict(json.load(f))
    if model_config.model_type == "simplified_transformer":
        model = SimplifiedBertBasedModel.TimelineRegressor(model_config)
    if model_config.model_type == "lstm":
        model = LSTMBasedModel.BiLSTMRegressor(model_config)
    model.load_state_dict(torch.load(f"{directory}/model_state_dict.pth", map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(directory + "/tokenizer")
    model.tokenizer = tokenizer
    return model, tokenizer

def train_and_evaluate_model_with_parameters(config):
    # Log the configuration to a file with the run date and time
    log_entry = f"\nRun Date and Time: {datetime.datetime.now().isoformat()}\nConfig: {config.get_config_as_string()}\n"
    with open("runs.txt", "a") as log_file:
        log_file.write(log_entry)

    if config.model_type == "simplified_transformer":
        model = SimplifiedBertBasedModel.TimelineRegressor(config)
    elif config.model_type == "full_transformer":
        model = BertBasedModel.TimelineRegressor(config)
    elif config.model_type == "lstm":
        model = LSTMBasedModel.BiLSTMRegressor(config)
    elif config.model_type == "closest_transformer":
        model = ClosestBertBasedModel.TimelineRegressor(config)

    # Load and preprocess data
    dataframe = load_i2b2_absolute_data()
    dataframe_test = load_i2b2_absolute_data(test_split=True)

    # Data analysis - print some statistics about the data
    print(f"Training data size: {len(dataframe)}")
    print(f"Testing data size: {len(dataframe_test)}")

    # Create datasets and train
    dataset = TimelineDataset(dataframe)
    dataset_test = TimelineDataset(dataframe_test)
    final_results = train(model, dataset, dataset_test, config)

    with open("runs.txt", "a") as log_file:
        log_entry = f"Run finished. Final results: {final_results}\n"
        log_file.write(log_entry)

    return final_results

if __name__ == '__main__':
    config = ModelConfig()
    config.training_hyperparameters["batch_size"] = 4
    config.model_type = "closest_transformer"
    # Train with the improved configuration
    results = train_and_evaluate_model_with_parameters(config)
    print(f"Final evaluation results: {results}")