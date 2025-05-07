class ModelConfig:
    def __init__(self):
        self.model_type = "simplified_transformer" # simplified_transformer, full_transformer
        self.simplified_transformer_config = {
            "model_name": "answerdotai/ModernBERT-base", # answerdotai/ModernBERT-base, emilyalsentzer/Bio_ClinicalBERT
            "pooling_strategy": "mean", # mean, max, cls [use cls token instead of event embeddings]
            "handle_too_long_text": "error", # error, cut
            "mark_events": True, # add <event> markers around te event: True, False
        }
