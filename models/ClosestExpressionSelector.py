import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

from models.model_config import ModelConfig


class ClosestExpressionSelector(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = AutoConfig.from_pretrained(model_config.closest_expression_selector_config["model_name"])
        self.encoder = AutoModel.from_pretrained(model_config.closest_expression_selector_config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.closest_expression_selector_config["model_name"])

        hidden_size = self.config.hidden_size
        intermediate_size = 128
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 2),  # Predict start and end only
            nn.Softmax(dim=1)  # Apply softmax to the output
        )

    def forward(self, row):
        """
        Forward pass of the model.

        Args:
            row (dict): A dictionary containing the input data.

        Returns:
            torch.Tensor: The output of the model.
        """
        text = row['text']
        target = row['labels']
        event_start_char = row['start_char']
        time_start_char = row['expression_char_start']

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.encoder(**inputs.to(self.encoder.device))
        last_hidden_state = outputs.last_hidden_state

        # Find token indices for event and time expressions
        event_token_idx = [inputs[i].char_to_token(int(event_start_char[i])) for i in range(len(text))]
        time_token_idx = [inputs[i].char_to_token(int(time_start_char[i])) for i in range(len(text))]

        # Gather embeddings
        event_embeddings = last_hidden_state[range(len(text)), event_token_idx, :]
        time_embeddings = last_hidden_state[range(len(text)), time_token_idx, :]

        # Concatenate and classify
        concat_embeddings = torch.cat([event_embeddings, time_embeddings], dim=1)
        logits = self.classifier(concat_embeddings)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        if target is None:
            loss = None
        else:
            loss = loss_fn(logits, target.long())

        return {"loss": loss, "logits": logits}
