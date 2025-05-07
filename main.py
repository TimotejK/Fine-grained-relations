import argparse
import os

import wandb
from transformers import AutoTokenizer

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from models import BertBasedModel
from train import train_bert_model


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
        # emilyalsentzer/Bio_ClinicalBERT
        model_name = 'bert-base-uncased'
        model = BertBasedModel.TimelineRegressor(model_name=model_name)
        dataframe = load_i2b2_absolute_data(test_split=False)
        dataframe_test = load_i2b2_absolute_data(test_split=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = TimelineDataset(dataframe, tokenizer)
        dataset_test = TimelineDataset(dataframe_test, tokenizer)
        train_bert_model.train(model, dataset, dataset_test)


if __name__ == "__main__":
    main()
