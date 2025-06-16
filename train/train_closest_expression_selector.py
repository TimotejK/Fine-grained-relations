import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass

from data_loaders.dataset import SelectingTimeExressionsDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from models.ClosestExpressionSelector import ClosestExpressionSelector
from models.model_config import ModelConfig


@dataclass
class DataCollatorForSelector:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        texts = [f["row"]["text"] for f in features]
        start_chars = [f["row"]["start_char"] for f in features]
        expression_chars = [f["row"]["expression_char_start"] for f in features]
        labels = torch.tensor([f["row"]["closest"] for f in features], dtype=torch.long)

        return { "row": {
                "text": texts,
                "start_char": start_chars,
                "expression_char_start": expression_chars,
                "labels": labels,
            }
        }

def compute_metrics(pred):
    from sklearn.metrics import precision_recall_fscore_support
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) if hasattr(pred.predictions, "argmax") else pred.predictions
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# --------------------------
# Training Script
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    df = load_i2b2_absolute_data()
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = SelectingTimeExressionsDataset(train_df)
    val_dataset = SelectingTimeExressionsDataset(val_df)

    # Model config
    model_config = ModelConfig()
    model = ClosestExpressionSelector(model_config).to(device)

    tokenizer = model.tokenizer
    collator = DataCollatorForSelector(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=5,
        optim="adamw_torch_fused",
        # weight_decay=0.001,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
        no_cuda=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()
    torch.save(model, "results/closest_expression_selector_model.pt")


if __name__ == "__main__":
    main()
