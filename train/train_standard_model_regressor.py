import gc
import json
import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import datetime
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assume these imports from your existing code
from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data


class SimpleTimelineModel(nn.Module):
    """Timeline regression model using AutoModelForSequenceClassification"""

    def __init__(self, model_name: str = "bert-base-uncased", mark_events: bool = True):
        super().__init__()
        self.model_name = model_name
        self.mark_events = mark_events

        # Load tokenizer first to handle special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens for event marking if needed
        if self.mark_events:
            special_tokens = ["<event>", "</event>"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # Load model for regression (3 outputs: start, end, duration)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # 3 outputs: start, end, duration
            problem_type="regression"  # Important for regression!
        )

        # Resize embeddings if we added special tokens
        if self.mark_events:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.label_scaling_factor = 1000.0  # Based on your original code

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


def get_closest_time_expression_by_distance(text, start_char_index, end_char_index, temporal_expressions):
    """Find closest temporal expression by character distance"""
    best_score = 0
    best_expression = None
    if isinstance(temporal_expressions, str):
        temporal_expressions = json.loads(temporal_expressions)
    for expression in temporal_expressions:

        score = 100000 - min(
            abs(expression["start"] - end_char_index),
            abs(expression["end"] - start_char_index)
        )

        if score >= best_score:
            best_score = score
            best_expression = expression

    if best_expression is None:
        print(f"No valid temporal expression found in text: {text}")
        return 0  # Default if no temporal expression found
    else:
        return best_expression['value_minutes']


def preprocess_labels_with_temporal_context(
        labels,
        texts,
        start_chars,
        end_chars,
        temporal_expressions_list,
        admission_date_minutes_list,
        label_scaling_factor
):
    """
    Preprocess labels to be relative to closest temporal expression
    Similar to the original model's preprocessing logic
    """
    processed_labels = []
    closest_time_minutes_list = []

    for i, (text, start_char, end_char, temporal_expressions, admission_date_minutes) in enumerate(
            zip(texts, start_chars, end_chars, temporal_expressions_list, admission_date_minutes_list)
    ):
        # Get closest temporal expression
        closest_time_minutes = get_closest_time_expression_by_distance(
            text, start_char, end_char, temporal_expressions
        )
        closest_time_minutes_list.append(closest_time_minutes)

        # Extract original labels for this sample
        start_label = labels[i][0].item() if torch.is_tensor(labels[i][0]) else labels[i][0]
        end_label = labels[i][1].item() if torch.is_tensor(labels[i][1]) else labels[i][1]
        duration_label = labels[i][2].item() if torch.is_tensor(labels[i][2]) else labels[i][2]

        # Convert start and end labels to be relative to closest time expression
        # (duration remains unchanged as it's not relative to time)
        adjusted_start = start_label - closest_time_minutes
        adjusted_end = end_label - closest_time_minutes

        # Scale all labels
        scaled_labels = torch.tensor([
            adjusted_start / label_scaling_factor,
            adjusted_end / label_scaling_factor,
            duration_label / label_scaling_factor
        ], dtype=torch.float32)

        processed_labels.append(scaled_labels)

    return torch.stack(processed_labels), closest_time_minutes_list


def postprocess_predictions(
        predictions,
        closest_time_minutes_list,
        admission_date_minutes_list,
        label_scaling_factor
):
    """
    Convert predictions back to original scale and time reference
    """
    # Scale predictions back
    scaled_predictions = predictions * label_scaling_factor

    processed_predictions = []
    for i, (pred, closest_time, admission_time) in enumerate(
            zip(scaled_predictions, closest_time_minutes_list, admission_date_minutes_list)
    ):
        # Convert start and end back to admission-relative time
        start_pred = pred[0] - admission_time + closest_time
        end_pred = pred[1] - admission_time + closest_time
        duration_pred = pred[2]  # Duration doesn't need time adjustment

        processed_predictions.append([start_pred, end_pred, duration_pred])

    return np.array(processed_predictions)


class TimelineDatasetForTrainer(Dataset):
    """Dataset wrapper compatible with transformers Trainer with temporal preprocessing"""

    def __init__(self, original_dataset: TimelineDataset, tokenizer, max_length: int = 512, mark_events: bool = True):
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mark_events = mark_events
        self.label_scaling_factor = 1000.0

        # Pre-compute all data to avoid repeated processing
        self._preprocess_all_data()

    def _preprocess_all_data(self):
        """Preprocess all data once during initialization"""
        self.processed_data = []

        texts = []
        start_chars = []
        end_chars = []
        temporal_expressions_list = []
        admission_dates = []
        raw_labels = []

        # Collect all data first
        for idx in range(len(self.original_dataset)):
            item = self.original_dataset[idx]

            texts.append(item['text'])
            start_chars.append(int(item['start_char']))
            end_chars.append(int(item['end_char']))
            temporal_expressions_list.append(item['temporal_expressions'])
            admission_dates.append(float(item['admission_date_minutes']))

            # Raw labels relative to admission date
            raw_labels.append(torch.tensor([
                float(item['start_time_minutes']),
                float(item['end_time_minutes']),
                float(item['duration_minutes'])
            ], dtype=torch.float32))

        # Preprocess all labels at once
        processed_labels, closest_times = preprocess_labels_with_temporal_context(
            raw_labels, texts, start_chars, end_chars,
            temporal_expressions_list, admission_dates, self.label_scaling_factor
        )

        # Store processed data
        for idx in range(len(texts)):
            # Prepare text with event markers
            text = texts[idx]
            if self.mark_events:
                start_char = start_chars[idx]
                end_char = end_chars[idx]
                # Ensure indices are within bounds
                start_char = max(0, min(start_char, len(text)))
                end_char = max(start_char, min(end_char, len(text)))

                text = (text[:start_char] +
                        "<event>" +
                        text[start_char:end_char] +
                        "</event>" +
                        text[end_char:])

            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors=None  # Return lists/arrays, not tensors
            )

            self.processed_data.append({
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'labels': processed_labels[idx],
                # Store metadata as primitive types, not tensors
                'closest_time_minutes': float(closest_times[idx]),
                'admission_date_minutes': float(admission_dates[idx]),
            })

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


class CustomDataCollator:
    """Custom data collator that properly handles metadata"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Separate the main features from metadata
        batch = {}
        metadata_keys = ['closest_time_minutes', 'admission_date_minutes']

        # Handle input_ids and attention_mask
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])

        # Handle metadata
        for key in metadata_keys:
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float32)

        return batch


class TimelineTrainer(Trainer):
    """Custom trainer that handles temporal preprocessing for evaluation"""

    def __init__(self, *args, **kwargs):
        self.label_scaling_factor = 1000.0
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        """Custom loss computation"""
        labels = inputs.pop("labels")
        # Remove metadata from inputs before passing to model
        metadata_keys = ['closest_time_minutes', 'admission_date_minutes']
        metadata = {}
        for key in metadata_keys:
            if key in inputs:
                metadata[key] = inputs.pop(key)

        outputs = model(**inputs)

        # Use L1Loss as in original code
        loss_fn = nn.L1Loss()
        loss = loss_fn(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step that preserves metadata for postprocessing"""
        # Extract metadata before removing from inputs
        closest_times = inputs.pop("closest_time_minutes", None)
        admission_times = inputs.pop("admission_date_minutes", None)

        # Standard prediction step
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        # Store metadata for later use in compute_metrics
        if not hasattr(self, '_prediction_metadata'):
            self._prediction_metadata = []

        if closest_times is not None:
            if torch.is_tensor(closest_times):
                closest_times = closest_times.cpu().numpy()
            if torch.is_tensor(admission_times):
                admission_times = admission_times.cpu().numpy()

            batch_size = len(closest_times)
            for i in range(batch_size):
                self._prediction_metadata.append({
                    'closest_time_minutes': float(closest_times[i]),
                    'admission_date_minutes': float(admission_times[i])
                })

        return loss, logits, labels


def compute_metrics(eval_pred):
    """Compute evaluation metrics with proper postprocessing"""
    predictions, labels = eval_pred

    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get trainer instance to access metadata (this is a bit hacky but works)
    # In a real implementation, you'd want to pass metadata more cleanly
    trainer = None
    for obj in gc.get_objects():
        if isinstance(obj, TimelineTrainer) and hasattr(obj, '_prediction_metadata'):
            trainer = obj
            break

    if trainer and hasattr(trainer, '_prediction_metadata'):
        metadata = trainer._prediction_metadata[-len(predictions):]  # Get last batch's metadata

        closest_times = [m['closest_time_minutes'] for m in metadata]
        admission_times = [m['admission_date_minutes'] for m in metadata]

        # Postprocess predictions to original scale
        processed_predictions = postprocess_predictions(
            predictions, closest_times, admission_times, trainer.label_scaling_factor
        )

        # Also postprocess labels for fair comparison
        processed_labels = postprocess_predictions(
            labels, closest_times, admission_times, trainer.label_scaling_factor
        )

        predictions = processed_predictions
        labels = processed_labels

    # Calculate metrics for each output (start, end, duration)
    metrics = {}
    output_names = ['start', 'end', 'duration']

    for i, name in enumerate(output_names):
        pred_i = predictions[:, i]
        label_i = labels[:, i]

        mae = mean_absolute_error(label_i, pred_i)
        mse = mean_squared_error(label_i, pred_i)
        rmse = np.sqrt(mse)

        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_mse'] = mse
        metrics[f'{name}_rmse'] = rmse

    # Overall metrics
    overall_mae = mean_absolute_error(labels.flatten(), predictions.flatten())
    overall_rmse = np.sqrt(mean_squared_error(labels.flatten(), predictions.flatten()))

    metrics['overall_mae'] = overall_mae
    metrics['overall_rmse'] = overall_rmse

    return metrics


def train_timeline_model():
    """Main training function"""
    # Set environment variable to suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configuration
    model_name = "answerdotai/ModernBERT-base"  # Can be changed to other models
    output_dir = "./results-standard-regressor"
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    dataframe = load_i2b2_absolute_data()
    dataframe_test = load_i2b2_absolute_data(test_split=True)

    print(f"Training data size: {len(dataframe)}")
    print(f"Testing data size: {len(dataframe_test)}")

    # Initialize model
    print("Initializing model...")
    model = SimpleTimelineModel(model_name=model_name, mark_events=True)
    tokenizer = model.tokenizer

    # Create datasets with temporal preprocessing
    print("Creating datasets...")
    train_dataset = TimelineDatasetForTrainer(
        TimelineDataset(dataframe),
        tokenizer,
        max_length=512,
        mark_events=True
    )

    eval_dataset = TimelineDatasetForTrainer(
        TimelineDataset(dataframe_test),
        tokenizer,
        max_length=512,
        mark_events=True
    )

    # Use custom data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # Training arguments - using recommended values for regression tasks
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,

        # Training hyperparameters
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        # Optimizer settings
        learning_rate=2e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,

        # Scheduler
        warmup_ratio=0.1,
        lr_scheduler_type="linear",

        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="overall_mae",
        greater_is_better=False,

        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to=["wandb"],  # Disable wandb for now

        # Other settings - FIXED: Reduce num_workers to avoid forking issues
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        remove_unused_columns=False,
        push_to_hub=False,

        # Reproducibility
        seed=42,
        data_seed=42,
    )

    # Initialize custom trainer
    trainer = TimelineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving model...")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print("Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Save evaluation results
    with open(f"{output_dir}/final_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    return eval_results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run training
    results = train_timeline_model()
    print(f"Training completed. Final results: {results}")