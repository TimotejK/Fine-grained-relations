import argparse
import os

import numpy as np
import torch

import wandb
from transformers import AutoTokenizer

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from data_loaders.preprocessing import preprocess
from evaluation.evaluate_llm_prompting import evaluate_all_llms
from models import BertBasedModel
from models.model_config import ModelConfig


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Main script to run specific modules.")
    parser.add_argument(
        "--script",
        type=str,
        choices=["train_default_bert_model", "train_bert_model", "preprocess", "finetune_llm", "prompting_llm",
                 "train_lstm_model", "train_time_finder"],
        help="The script to run. Currently supported: train_bert_model, preprocess, finetune_llm, prompting_llm"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B",
        help=""
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
        # df = load_i2b2_absolute_data(test_split=False)
        #
        # df = df.apply(preprocess, axis=1)
        # torch.save(df, "data/i2b2_train_absolute_preprocessed.pt")

        df = load_i2b2_absolute_data(test_split=True)

        df = df.apply(preprocess, axis=1)
        torch.save(df, "data/i2b2_test_absolute_preprocessed.pt")

    if args.script == "train_default_bert_model":
        from train.train_standard_model_regressor import train_timeline_model
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run training
        results = train_timeline_model()
        print(f"Training completed. Final results: {results}")
    elif args.script == "train_bert_model":
        from train import train_bert_model
        from train.train_bert_model import train_and_evaluate_model_with_parameters
        print("Running train_bert_model...")
        config = ModelConfig()
        config.model_type = "closest_transformer"

        # Adjust hyperparameters for more stable training
        config.simplified_transformer_config["individually_train_regressor_number"] = -1  # Train all regressors
        config.simplified_transformer_config["predicted_minutes_scaling_factor"] = 1  # should improve stability
        config.training_hyperparameters["seed"] = 42  # Set a fixed seed for reproducibility
        config.training_hyperparameters["weight_decay"] = 0.01  # Stronger regularization
        config.training_hyperparameters["batch_size"] = 16  # Adjust batch size as needed
        config.training_hyperparameters["epochs"] = 20  # More epochs with early stopping

        config.training_hyperparameters["learning_rate"] = 2e-5  # Lower learning rate
        train_and_evaluate_model_with_parameters(config)
        config.training_hyperparameters["learning_rate"] = 2e-3  # Higher learning rate
        train_and_evaluate_model_with_parameters(config)
        config.training_hyperparameters["learning_rate"] = 2e-1  # Very high learning rate
        train_and_evaluate_model_with_parameters(config)

        # config.simplified_transformer_config["pooling_strategy"] = "mean"
        # config.simplified_transformer_config["model_name"] = "Simonlee711/Clinical_ModernBERT"
        # config.training_hyperparameters["learning_rate"] = 2e-5  # Lower learning rate
        # train_and_evaluate_model_with_parameters(config)
        #
        # config.simplified_transformer_config["pooling_strategy"] = "max"
        # config.simplified_transformer_config["model_name"] = "Simonlee711/Clinical_ModernBERT"
        # config.training_hyperparameters["learning_rate"] = 2e-5  # Lower learning rate
        # train_and_evaluate_model_with_parameters(config)


    if args.script == "train_time_finder":
        from train import train_closest_expression_selector
        print("Running train_bert_model...")
        train_closest_expression_selector.main()

    if args.script == "train_lstm_model":
        from train import train_bert_model
        from train.train_bert_model import train_and_evaluate_model_with_parameters
        print("Running train_bert_model...")
        config = ModelConfig()
        # Adjust hyperparameters for more stable training
        config.model_type = "lstm"  # Train all regressors
        config.simplified_transformer_config["individually_train_regressor_number"] = -1  # Train all regressors
        config.simplified_transformer_config["predicted_minutes_scaling_factor"] = 10000  # should improve stability
        config.training_hyperparameters["seed"] = 42  # Set a fixed seed for reproducibility
        config.training_hyperparameters["learning_rate"] = 2e-5  # Lower learning rate
        config.training_hyperparameters["weight_decay"] = 0.01  # Stronger regularization
        config.training_hyperparameters["batch_size"] = 16  # Adjust batch size as needed
        config.training_hyperparameters["epochs"] = 20  # More epochs with early stopping

        train_and_evaluate_model_with_parameters(config)
        config.training_hyperparameters["learning_rate"] = 2e-3  # Higher learning rate
        train_and_evaluate_model_with_parameters(config)
        config.training_hyperparameters["learning_rate"] = 2e-1  # Very high learning rate
        train_and_evaluate_model_with_parameters(config)

    if args.script == "finetune_llm":
        from llm_finetuning.finetuning import finetune_on_i2b2
        finetune_on_i2b2(args.model)

    if args.script == "prompting_llm":
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

        evaluate_all_llms()


if __name__ == "__main__":
    main()
