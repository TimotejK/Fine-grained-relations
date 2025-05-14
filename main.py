import argparse
import os

import torch

import wandb
from transformers import AutoTokenizer

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from data_loaders.preprocessing import preprocess
from models import BertBasedModel
from models.model_config import ModelConfig
from train import train_bert_model
from train.train_bert_model import train_and_evaluate_model_with_parameters


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Main script to run specific modules.")
    parser.add_argument(
        "--script",
        type=str,
        choices=["train_bert_model", "preprocess"],
        help="The script to run. Currently supported: train_bert_model"
    )

    # Parse arguments
    args = parser.parse_args()

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    # Execute the appropriate script based on the argument
    if args.script == "preprocess":
        # start ollama on hpc
        import threading
        import subprocess
        import time

        def run_ollama_serve():
            subprocess.Popen(["ollama", "serve"])

        thread = threading.Thread(target=run_ollama_serve)
        thread.start()
        time.sleep(5)
        subprocess.run(["ollama", "pull", "gemma3:27b"])
        df = load_i2b2_absolute_data(test_split=False)

        df = df.apply(preprocess, axis=1)
        torch.save(df, "data/i2b2_train_absolute_preprocessed.pt")

        df = load_i2b2_absolute_data(test_split=True)

        df = df.apply(preprocess, axis=1)
        torch.save(df, "data/i2b2_test_absolute_preprocessed.pt")

    if args.script == "train_bert_model":
        print("Running train_bert_model...")
        config = ModelConfig()
        train_and_evaluate_model_with_parameters(config)

        config.simplified_transformer_config["pooling_strategy"] = "mean"
        config.simplified_transformer_config["model_name"] = "Simonlee711/Clinical_ModernBERT"
        train_and_evaluate_model_with_parameters(config)

        config.simplified_transformer_config["pooling_strategy"] = "max"
        config.simplified_transformer_config["model_name"] = "Simonlee711/Clinical_ModernBERT"
        train_and_evaluate_model_with_parameters(config)


if __name__ == "__main__":
    main()
