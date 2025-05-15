import json

import torch
import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import evaluate_temporal_predictions, compute_metrics
from models import BertBasedModel, SimplifiedBertBasedModel
from models.model_config import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps") if torch.backends.mps.is_available() else device

def evaluate(model, dataloader, simplified_model) -> (float, dict):
    model.eval()
    total_eval_loss = 0
    predicted_starts = []
    predicted_ends = []
    predicted_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []

    with torch.no_grad():
        for batch in dataloader:
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

            admission_times = batch['admission_date_minutes']
            if simplified_model:
                outputs = model(
                    text=batch['text'],
                    labels=labels,
                    start_char_index=batch['start_char'],
                    end_char_index=batch['end_char']
                )
                total_eval_loss += outputs['loss'].item()
                start_minutes = (outputs['predictions'][:, 0].cpu() + admission_times).tolist()
                end_minutes = (outputs['predictions'][:, 1].cpu() + admission_times).tolist()
                duration_minutes = (outputs['predictions'][:, 2].cpu() + admission_times).tolist()
                start_minutes = [(t, t - 60, t+60) for t in start_minutes] # TODO implement better lower and upper bound guess
                end_minutes = [(t, t - 60, t+60) for t in end_minutes]
                duration_minutes = [(t, t - 60, t+60) for t in duration_minutes]
            else:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=labels
                )
                start_minutes, end_minutes = dataloader.dataset.convert_prediction_to_relative_time_minutes(outputs)
                start_minutes = [(float(t+at),float(l+at),float(u+at)) for (t,l,u), at in zip(zip(*start_minutes), admission_times)]
                end_minutes = [(float(t+at),float(l+at),float(u+at)) for (t,l,u), at in zip(zip(*end_minutes), admission_times)]

            predicted_starts += start_minutes
            predicted_ends += end_minutes
            predicted_durations += duration_minutes
            gold_starts += [(float(t),float(l),float(u)) for t,l,u in zip(batch["start_time_minutes"], batch["start_lower_minutes"], batch["start_upper_minutes"])]
            gold_ends += [(float(t),float(l),float(u)) for t,l,u in zip(batch["end_time_minutes"], batch["end_lower_minutes"], batch["end_upper_minutes"])]
            gold_durations += [(float(t),float(l),float(u)) for t,l,u in zip(batch["duration_minutes"], batch["duration_lower_minutes"], batch["duration_upper_minutes"])]
            eval_metrics = compute_metrics(predicted_starts, predicted_ends, predicted_durations, gold_starts, gold_ends, gold_durations)
    print("End of epoch results on eval:")

    print(eval_metrics)

    avg_eval_loss = total_eval_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    return avg_eval_loss, eval_metrics

def train(model, dataset, dataset_test, config, project_name="timeline_training"):

    import wandb
    model.to(device)

    simplified_model = config.model_type == "simplified_transformer"
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

    wandb.init(project=project_name, config={
        "model_name": model.encoder.__class__.__name__,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    })

    wandb.watch(model, log="all", log_freq=10)
    wandb.config.update(config.to_dict())  # Log the model configuration to wandb

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
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

            outputs = model(
                text=batch['text'],
                labels=labels,
                start_char_index=batch['start_char'],
                end_char_index=batch['end_char']
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Log training loss
        scheduler.step()
        wandb.log({"epoch": epoch + 1, "train_loss": total_loss / len(dataloader), "lr": scheduler.get_last_lr()[0]})
        wandb.log({"epoch": epoch + 1, "train_loss": total_loss / len(dataloader)})
        # Evaluate after every epoch
        eval_loss, eval_metrics = evaluate(model, dataloader_test, simplified_model=simplified_model)

        with open("runs.txt", "a") as log_file:
            log_entry = f"{eval_metrics}\n"
            log_file.write(log_entry)
        wandb.log({"epoch": epoch + 1, "eval_loss": eval_loss, "eval_metrics": eval_metrics})
        print({"epoch": epoch + 1, "eval_loss": eval_loss, "eval_metrics": eval_metrics})

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
    save_model(model, model.tokenizer, f"{model.encoder.__class__.__name__}_{config.model_type}")
    wandb.finish()
    return eval_metrics

def save_model(model, tokenizer, model_name):
    torch.save(model.state_dict(), f"{model_name}_state_dict.pth")
    tokenizer.save_pretrained(model_name)
    with open(f"{model_name}_config.json", "w") as f:
        json.dump(model.config.to_dict(), f)

def load_model(model_name, device=torch.device('cpu')):
    with open(f"{model_name}_config.json", "r") as f:
        config = ModelConfig.from_dict(json.load(f))
    model = BertBasedModel.TimelineRegressor(model_name=config.model_type)
    model.load_state_dict(torch.load(f"{model_name}_state_dict.pth", map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        model = BertBasedModel.TimelineRegressor(model_name=model_name)
    dataframe = load_i2b2_absolute_data()
    dataframe_test = load_i2b2_absolute_data(test_split=True)
    dataset = TimelineDataset(dataframe)
    dataset_test = TimelineDataset(dataframe_test)
    final_results = train(model, dataset, dataset_test, config)
    with open("runs.txt", "a") as log_file:
        log_entry = f"Run finished. Final results: {final_results}\n"
        log_file.write(log_entry)

if __name__ == '__main__':
    # emilyalsentzer/Bio_ClinicalBERT
    config = ModelConfig()
    train_and_evaluate_model_with_parameters(config)
    # train_and_evaluate_model_with_parameters('bert-base-uncased', True)
    # train_and_evaluate_model_with_parameters('bert-base-uncased', False)