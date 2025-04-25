import torch
from transformers import AdamW, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import evaluate_temporal_predictions, compute_metrics
from models import BertBasedModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader) -> (float, dict):
    model.eval()
    total_eval_loss = 0
    predicted_starts = []
    predicted_ends = []
    gold_starts = []
    gold_ends = []

    with torch.no_grad():
        for batch in dataloader:
            labels = (
                batch['start_unit'].to(device),
                batch['start_value'].to(device),
                batch['end_unit'].to(device),
                batch['end_value'].to(device),
                batch['duration_unit'].to(device),
                batch['duration_value'].to(device)
            )

            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=labels
            )
            admission_times = batch['admission_date_minutes'].to(device)
            start_minutes, end_minutes = dataloader.dataset.convert_prediction_to_relative_time_minutes(outputs)
            start_minutes = [(float(t+at),float(l+at),float(u+at)) for (t,l,u), at in zip(zip(*start_minutes), admission_times)]
            end_minutes = [(float(t+at),float(l+at),float(u+at)) for (t,l,u), at in zip(zip(*end_minutes), admission_times)]

            predicted_starts += start_minutes
            predicted_ends += end_minutes
            gold_starts += [(float(t),float(l),float(u)) for t,l,u in zip(batch["start_time_minutes"], batch["start_lower_minutes"], batch["start_upper_minutes"])]
            gold_ends += [(float(t),float(l),float(u)) for t,l,u in zip(batch["end_time_minutes"], batch["end_lower_minutes"], batch["end_upper_minutes"])]
    eval_metrics = compute_metrics(predicted_starts, predicted_ends, gold_starts, gold_ends)
    print("End of epoch results on eval:")

    print(eval_metrics)

    avg_eval_loss = total_eval_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    return avg_eval_loss, eval_metrics

def train(model, dataset, epochs=3, batch_size=16, lr=2e-5, project_name="timeline_training"):

    import wandb
    wandb.init(project=project_name, config={
        "model_name": model.encoder.__class__.__name__,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr
    })
    wandb.watch(model, log="all", log_freq=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            labels = (
                batch['start_unit'].to(device),
                batch['start_value'].to(device),
                batch['end_unit'].to(device),
                batch['end_value'].to(device),
                batch['duration_unit'].to(device),
                batch['duration_value'].to(device)
            )

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=labels
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log training loss
        wandb.log({"epoch": epoch + 1, "train_loss": total_loss / len(dataloader)})
        # Evaluate after every epoch

        eval_loss, eval_metrics = evaluate(model, dataloader)
        wandb.log({"epoch": epoch + 1, "eval_loss": eval_loss, "eval_metrics": eval_metrics})

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    wandb.finish()
if __name__ == '__main__':
    # emilyalsentzer/Bio_ClinicalBERT
    model_name = 'bert-base-uncased'
    model = BertBasedModel.TimelineRegressor(model_name=model_name)
    dataframe = load_i2b2_absolute_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TimelineDataset(dataframe, tokenizer)
    train(model, dataset)