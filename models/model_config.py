class ModelConfig:
    def __init__(self):
        self.model_type = "simplified_transformer" # simplified_transformer, full_transformer
        self.simplified_transformer_config = {
            "model_name": "answerdotai/ModernBERT-base", # answerdotai/ModernBERT-base, emilyalsentzer/Bio_ClinicalBERT
            "pooling_strategy": "mean", # mean, max, cls [use cls token instead of event embeddings]
            "handle_too_long_text": "error", # error, cut
            "mark_events": True, # add <event> markers around te event: True, False
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
        ]
        return "\n".join(config_lines)
