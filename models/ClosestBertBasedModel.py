import json

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

        # self.closest_expression_selector = torch.load("models/closest_expression_selector_model.pt", map_location=self.encoder.device)

        hidden_size = self.config.hidden_size
        intermediate_size = 128

        # Regressor for start and end (first two values)
        self.start_end_regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 2)  # Predict start and end only
        )

        # Separate regressor for duration (third value) with softplus activation
        self.duration_regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 1)  # Predict duration only
        )

    def softplus(self, x):
        """Softplus activation: ln(1 + e^x)"""
        return torch.log(1 + torch.exp(x))

    def tokenize(self, text, start_char, end_char):
        start_char_copy = [int(s) for s in start_char]
        end_char_copy = [int(e) for e in end_char]
        if self.model_config.simplified_transformer_config["mark_events"]:
            modified_texts = []

            for i, t in enumerate(text):
                event_marker_start = "<event>"
                event_marker_end = "</event>"
                t = (
                        t[:start_char_copy[i]]
                        + event_marker_start
                        + t[start_char_copy[i]:end_char_copy[i]]
                        + event_marker_end
                        + t[end_char_copy[i]:]
                )
                start_char_copy[i] += len(event_marker_start)
                end_char_copy[i] += len(event_marker_start)
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
            start_tokens.append(inputs.char_to_token(i, start_char_copy[i]))
            end_tokens.append(inputs.char_to_token(i, end_char_copy[i]))

        return {**inputs, "start_token": start_tokens, "end_token": end_tokens, "modified_text": text}

    def get_closest_time_expression(self, text, start_char_index, end_char_index, temporal_expressions):
        best_score = 0
        best_expression = None
        for expression in temporal_expressions:
            score = self.closest_expression_selector({"labels": None,
                                                      "text": text,
                                                     "start_char": start_char_index,
                                                      "expression_char_start": expression["expression_char_start"]})
            if score >= best_score:
                best_score = score
                best_expression = expression
        return best_expression['value_minutes']

    def get_closest_time_expression_by_distance(self, text, start_char_index, end_char_index, temporal_expressions):
        best_score = 0
        best_expression = None
        for expression in temporal_expressions:
            score = 100000 - min(abs(expression["start"] - end_char_index),
                abs(expression["end"] - start_char_index))
            if score >= best_score:
                best_score = score
                best_expression = expression
        if best_expression is None:
            return None
        else:
            return best_expression['value_minutes']

    def forward(self, text, labels=None, start_char_index=None, end_char_index=None, temporal_expressions=None, admission_date_minutes=0):
        device = self.encoder.device
        temporal_expressions = [json.loads(expression) for expression in temporal_expressions]
        tokenized = self.tokenize(text, start_char_index, end_char_index)
        outputs = self.encoder(input_ids=tokenized['input_ids'].to(device),
                               attention_mask=tokenized['attention_mask'].to(device))
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

        # Predict start and end values
        start_end_outputs = self.start_end_regressor(embeddings)

        # Predict duration with softplus activation to ensure positive values
        duration_raw = self.duration_regressor(embeddings)
        duration_outputs = self.softplus(duration_raw)

        # Combine outputs: [start, end, duration]
        regression_outputs = torch.cat([start_end_outputs, duration_outputs], dim=1)

        # Optionally calculate loss if labels are provided
        label_scaling_factor = self.model_config.simplified_transformer_config["predicted_minutes_scaling_factor"] # based on paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9207839

        # closest_time_minutes = [self.get_closest_time_expression(text, start_char_index, end_char_index, temporal_expressions)
        #                         for text, start_char_index, end_char_index, temporal_expressions in zip(text, start_char_index, end_char_index, temporal_expressions)
        #                         ]
        closest_time_minutes = [self.get_closest_time_expression_by_distance(text, start_char_index, end_char_index, temporal_expressions)
                                for text, start_char_index, end_char_index, temporal_expressions in zip(text, start_char_index, end_char_index, temporal_expressions)
                                ]
        loss = None
        if labels is not None:
            loss_fn = nn.L1Loss()
            # At this point, labels are relative to the admission time
            # convert labels to relative to closest time expression
            labels = [torch.tensor([label + admission_date_minutes[i] - closest_time_minutes[i] for i, label in enumerate(l)]) if j < 2 else l for j, l in enumerate(labels)]

            labels = [label / label_scaling_factor for label in labels]

            # Only compute loss for the first value
            if self.model_config.simplified_transformer_config['individually_train_regressor_number'] >= 0:
                regressor_index = self.model_config.simplified_transformer_config['individually_train_regressor_number']
                loss = loss_fn(regression_outputs[:, regressor_index], labels[regressor_index].t().float().to(device))
            else:
                loss = loss_fn(regression_outputs, torch.stack(labels).t().float().to(device))
        if loss is not None and torch.isnan(loss):
            print(loss)
            print(regression_outputs)
            print(labels)

        predictions = regression_outputs * label_scaling_factor
        predictions = predictions.cpu().detach()
        predictions[:, 0] -= admission_date_minutes
        predictions[:, 1] -= admission_date_minutes
        predictions[:, 0] += torch.tensor(closest_time_minutes)
        predictions[:, 1] += torch.tensor(closest_time_minutes)
        return {
            "loss": loss,
            # TODO translate the predictions back to the original offset format
            "predictions": predictions
        }