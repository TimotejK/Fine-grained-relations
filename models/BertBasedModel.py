from transformers import AutoModel, AutoConfig
import torch.nn as nn


class TimelineRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_units=5):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size
        intermediate_size = 128

        # Classifiers for time units
        self.start_unit_classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, num_units)
        )
        self.end_unit_classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, num_units)
        )
        self.duration_unit_classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, num_units)
        )

        # Regressors for time values
        self.start_value_regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 1)
        )
        self.end_value_regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 1)
        )
        self.duration_value_regressor = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 1)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Predict units (classification)
        start_unit_logits = self.start_unit_classifier(cls_output)
        end_unit_logits = self.end_unit_classifier(cls_output)
        duration_unit_logits = self.duration_unit_classifier(cls_output)

        # Predict values (regression)
        start_value = self.start_value_regressor(cls_output)
        end_value = self.end_value_regressor(cls_output)
        duration_value = self.duration_value_regressor(cls_output)

        # Optionally calculate loss if labels provided
        loss = None
        if labels is not None:
            (start_unit_labels, start_value_labels, end_unit_labels, end_value_labels, duration_unit_labels, duration_value_labels) = labels

            loss_fn_cls = nn.CrossEntropyLoss()
            loss_fn_reg = nn.MSELoss()

            loss = (
                    loss_fn_cls(start_unit_logits, start_unit_labels) +
                    loss_fn_reg(start_value.squeeze(), start_value_labels.float()) +
                    loss_fn_cls(end_unit_logits, end_unit_labels) +
                    loss_fn_reg(end_value.squeeze(), end_value_labels.float()) +
                    loss_fn_cls(duration_unit_logits, duration_unit_labels) +
                    loss_fn_reg(duration_value.squeeze(), duration_value_labels.float())
            )

        return {
            "loss": loss,
            "start_unit_logits": start_unit_logits,
            "start_value": start_value,
            "end_unit_logits": end_unit_logits,
            "end_value": end_value,
            "duration_unit_logits": duration_unit_logits,
            "duration_value": duration_value
        }
