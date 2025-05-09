import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch.nn as nn

from models.model_config import ModelConfig


class TimelineRegressor(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = AutoConfig.from_pretrained(model_config.simplified_transformer_config["model_name"])
        self.encoder = AutoModel.from_pretrained(model_config.simplified_transformer_config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.simplified_transformer_config["model_name"])


        hidden_size = self.config.hidden_size
        intermediate_size = 128

        # Lightweight regression model
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 3)  # Predict start, end, and duration
        )

        # Removed individual regressors for simplicity

    def tokenize(self, text, start_char, end_char):

        if self.model_config.simplified_transformer_config["mark_events"]:
            modified_texts = []
            for i, t in enumerate(text):
                event_marker_start = "<event>"
                event_marker_end = "</event>"
                t = (
                        t[:start_char[i]]
                        + event_marker_start
                        + t[start_char[i]:end_char[i]]
                        + event_marker_end
                        + t[end_char[i]:]
                )
                start_char[i] += len(event_marker_start)
                end_char[i] += len(event_marker_start)
                modified_texts.append(t)
            text = modified_texts

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        if (self.model_config.simplified_transformer_config["handle_too_long_text"] == "error" and
                inputs["input_ids"].size()[1] >= self.tokenizer.model_max_length):
            raise ValueError("The provided text was too long and has been truncated.")

        start_tokens = []
        end_tokens = []
        for i in range(len(inputs["input_ids"])):
            start_tokens.append(inputs.char_to_token(i, start_char[i]))
            end_tokens.append(inputs.char_to_token(i, end_char[i]))

        return {**inputs, "start_token": start_tokens, "end_token": end_tokens, "modified_text": text}

    def forward(self, text, labels=None, start_char_index=None, end_char_index=None):
        device = self.encoder.device
        tokenized = self.tokenize(text, start_char_index, end_char_index)
        outputs = self.encoder(input_ids=tokenized['input_ids'].to(device), attention_mask=tokenized['attention_mask'].to(device))
        if self.model_config.simplified_transformer_config["pooling_strategy"] == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            # Extract embeddings for the tokens between the start and end indices
            token_embeddings = outputs.last_hidden_state
            embeddings = []
            for i in range(token_embeddings.size()[0]):
                event_embeddings = token_embeddings[i, tokenized['start_token'][i]:tokenized['end_token'][i], :]
                if self.model_config.simplified_transformer_config["pooling_strategy"] == "mean":
                    event_embeddings_pooled = torch.mean(event_embeddings, dim=0)
                elif self.model_config.simplified_transformer_config["pooling_strategy"] == "max":
                    event_embeddings_pooled, _ = torch.max(event_embeddings, dim=0)
                embeddings.append(event_embeddings_pooled)
                pass
            embeddings = torch.stack(embeddings)

        # Predict start, end, and duration in a single pass
        regression_outputs = self.regressor(embeddings)

        # Optionally calculate loss if labels are provided
        label_scaling_factor = 1000 # based on paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9207839
        loss = None
        if labels is not None:
            loss_fn = nn.L1Loss()
            labels = [label / label_scaling_factor for label in labels]
            loss = loss_fn(regression_outputs, torch.stack(labels).t().float().to(device))

        if loss is not None and torch.isnan(loss):
            print(loss)
            print(regression_outputs)
            print(labels)
        return {
            "loss": loss,
            "predictions": regression_outputs * label_scaling_factor
        }