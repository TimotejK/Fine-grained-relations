class ModelConfig:
    def __init__(self):
        self.model_type = "simplified_transformer" # simplified_transformer, full_transformer, lstm
        self.simplified_transformer_config = {
            "model_name": "answerdotai/ModernBERT-base", # answerdotai/ModernBERT-base, emilyalsentzer/Bio_ClinicalBERT, Simonlee711/Clinical_ModernBERT
            "pooling_strategy": "mean", # mean, max, cls [use cls token instead of event embeddings]
            "handle_too_long_text": "error", # error, cut
            "mark_events": True, # add <event> markers around te event: True, False
            "predicted_minutes_scaling_factor": 1, # divide the number of minutes by this value when computing loss to stabilize training
            "individually_train_regressor_number": -1
        }
        self.training_hyperparameters = {
            "learning_rate": 2e-5,
            "batch_size": 2,
            "epochs": 60,
            "seed": 42,
            "weight_decay": 0.01,
            "scheduler_config": {
                "step_size": 1, # every 10 epochs multiply learning rate by 0.1
                "gamma": 0.8
            }
        }

    def get_config_as_string(self):
        """
        Returns the model configuration as a formatted string for logging purposes.
        """
        config_lines = [
            f"Model Type: {self.model_type}",
            "Simplified Transformer Config:",
            f"  Model Name: {self.simplified_transformer_config['model_name']}",
            f"  Pooling Strategy: {self.simplified_transformer_config['pooling_strategy']}",
            f"  Handle Too Long Text: {self.simplified_transformer_config['handle_too_long_text']}",
            f"  Mark Events: {self.simplified_transformer_config['mark_events']}",
            "  Predicted Minutes Scaling Factor: {self.simplified_transformer_config['predicted_minutes_scaling_factor']}",
            "Training Hyperparameters:",
            f"  Learning Rate: {self.training_hyperparameters['learning_rate']}",
            f"  Batch Size: {self.training_hyperparameters['batch_size']}",
            f"  Epochs: {self.training_hyperparameters['epochs']}",
            f"  Seed: {self.training_hyperparameters['seed']}",
            f"  Weight Decay: {self.training_hyperparameters['weight_decay']}",
            "  Scheduler Config:",
            f"    Step Size: {self.training_hyperparameters['scheduler_config']['step_size']}",
            f"    Gamma: {self.training_hyperparameters['scheduler_config']['gamma']}",
        ]
        return "\n".join(config_lines)


    def to_dict(self):
        """
        Converts the model configuration into a dictionary format.
        """
        return {
            "model_type": self.model_type,
            "simplified_transformer_config": self.simplified_transformer_config,
            "training_hyperparameters": self.training_hyperparameters,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """
        Creates an instance of ModelConfig and sets its configuration from a dictionary.
    
        Args:
            config_dict (dict): A dictionary containing the model configuration.
    
        Returns:
            ModelConfig: An instance of ModelConfig with the configuration set.
        """
        instance = cls()
        if "model_type" in config_dict:
            instance.model_type = config_dict["model_type"]
        if "simplified_transformer_config" in config_dict:
            instance.simplified_transformer_config.update(config_dict["simplified_transformer_config"])
        if "training_hyperparameters" in config_dict:
            instance.training_hyperparameters.update(config_dict["training_hyperparameters"])
        return instance