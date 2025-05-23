import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence
from models.model_config import ModelConfig


class BiLSTMRegressor(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.embedding_dim = 300
        self.hidden_dim = 256
        self.num_layers = 2
        self.glove = GloVe(name='840B', dim=self.embedding_dim)
        self.word_to_idx = self.glove.stoi
        self.embedding = nn.Embedding.from_pretrained(self.glove.vectors, freeze=True)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=True, batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

    def tokenize(self, text, start_char, end_char):
        tokenized_sequences = []
        start_tokens = []
        end_tokens = []

        for i, t in enumerate(text):
            tokens = t.split()
            start = len(t[:start_char[i]].split())
            end = len(t[:end_char[i]].split())
            indices = [self.word_to_idx.get(tok.lower(), 0) for tok in tokens]
            tokenized_sequences.append(torch.tensor(indices))
            start_tokens.append(start)
            end_tokens.append(end)

        padded = pad_sequence(tokenized_sequences, batch_first=True)
        attention_mask = (padded != 0).long()
        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "start_token": start_tokens,
            "end_token": end_tokens,
            "modified_text": text
        }

    def forward(self, text, labels=None, start_char_index=None, end_char_index=None):
        device = next(self.parameters()).device
        tokenized = self.tokenize(text, start_char_index, end_char_index)

        inputs = tokenized["input_ids"].to(device)
        mask = tokenized["attention_mask"].to(device)

        embeddings = self.embedding(inputs)
        packed_output, _ = self.lstm(embeddings)

        pooled = []
        for i in range(len(inputs)):
            start = tokenized["start_token"][i]
            end = tokenized["end_token"][i]
            event_emb = packed_output[i, start:end, :]
            if self.model_config.simplified_transformer_config["pooling_strategy"] == "mean":
                pooled_vec = event_emb.mean(dim=0)
            elif self.model_config.simplified_transformer_config["pooling_strategy"] == "max":
                pooled_vec, _ = event_emb.max(dim=0)
            else:
                pooled_vec = event_emb[0]  # Fallback to first token
            pooled.append(pooled_vec)
        pooled = torch.stack(pooled)

        regression_outputs = self.regressor(pooled)

        label_scaling_factor = self.model_config.simplified_transformer_config["predicted_minutes_scaling_factor"]
        loss = None
        if labels is not None:
            loss_fn = nn.L1Loss()
            labels = [label / label_scaling_factor for label in labels]
            if self.model_config.simplified_transformer_config['individually_train_regressor_number'] >= 0:
                idx = self.model_config.simplified_transformer_config['individually_train_regressor_number']
                loss = loss_fn(regression_outputs[:, idx], labels[idx].t().float().to(device))
            else:
                loss = loss_fn(regression_outputs, torch.stack(labels).t().float().to(device))
        return {
            "loss": loss,
            "predictions": regression_outputs * label_scaling_factor
        }
