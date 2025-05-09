import argparse
import os

import wandb
from transformers import AutoTokenizer

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
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
        choices=["train_bert_model"],
        help="The script to run. Currently supported: train_bert_model"
    )

    # Parse arguments
    args = parser.parse_args()

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    # Execute the appropriate script based on the argument
    if args.script == "train_bert_model":
        print("Running train_bert_model...")
        config = ModelConfig()
        train_and_evaluate_model_with_parameters(config)
        config.simplified_transformer_config["pooling_strategy"] = "max"
        train_and_evaluate_model_with_parameters(config)


if __name__ == "__main__":
    main()
