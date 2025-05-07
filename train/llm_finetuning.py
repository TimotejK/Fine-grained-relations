import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import bitsandbytes as bnb

# LoRA integration
from peft import LoraConfig, get_peft_model


def load_i2b2_dataset(data_dir):
    """
    Load the i2b2 temporal relation dataset.

    Args:
        data_dir (str): Path to the i2b2 dataset.

    Returns:
        Dataset object containing training and validation data.
    """
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "train.json"),
        "validation": os.path.join(data_dir, "valid.json")
    })

    return dataset


class TemporalExtractionDataset(torch.utils.data.Dataset):
    """
    Dataset class for temporal relation extraction tasks.

    Converts i2b2 data into tokenized inputs and outputs for the language model.

    Args:
        data (list): List of data points from the dataset.
        tokenizer (AutoTokenizer): Tokenizer for the pre-trained LLM.

    """

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example["text"]
        label = f"Event happened {example['temporal_relation']} {example['admission_time']}."

        # Tokenize input and label
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                label,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

        inputs["labels"] = labels["input_ids"]
        return {key: torch.squeeze(val) for key, val in inputs.items()}


def build_lora_model(base_model_name):
    """
    Load LLaMA (or other models) and apply LoRA using bitsandbytes.

    Args:
        base_model_name (str): Name of the pre-trained model.

    Returns:
        LoRA-wrapped model and its tokenizer.
    """
    # Load Pretrained Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Fine-tune attention modules
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_temporal_extractor(model, tokenizer, dataset, output_dir, epochs=3, batch_size=4):
    """
    Train the temporal relation extraction model on the custom dataset.

    Args:
        model: Pre-trained and LoRA-fine-tuned model.
        tokenizer: Tokenizer for the model.
        dataset: Training and validation dataset.
        output_dir (str): Directory to save the trained model.
        epochs (int): Number of fine-tuning epochs.
        batch_size (int): Batch size for training.
    """
    # Tokenize dataset
    train_dataset = TemporalExtractionDataset(dataset["train"], tokenizer)
    val_dataset = TemporalExtractionDataset(dataset["validation"], tokenizer)

    # Define Trainer Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        fp16=True,  # Use mixed precision
        optim="adamw_torch"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Start Training
    trainer.train()

    # Save Model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # Define paths
    DATA_DIR = "./i2b2_data"  # Path to your i2b2 dataset folder
    OUTPUT_DIR = "./temporal_model"  # Path to save the fine-tuned model

    # Base pre-trained model (can be easily replaced)
    BASE_MODEL_NAME = "meta-llama/Llama-2-8b-hf"

    # Load dataset
    dataset = load_i2b2_dataset(DATA_DIR)

    # Load model and tokenizer with LoRA applied
    model, tokenizer = build_lora_model(BASE_MODEL_NAME)

    # Train the model
    train_temporal_extractor(model, tokenizer, dataset, OUTPUT_DIR, epochs=3, batch_size=4)

    print(f"Model successfully trained and saved in: {OUTPUT_DIR}")
