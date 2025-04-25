import numpy as np
from numpy.ma.core import argmax
from torch.utils.data import Dataset
import torch

class TimelineDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def convert_minutes_to_unit_value(self, minutes):
        minutes_in_unit = [365*24*60, 30*24*60, 24*60, 60, 1] # year, month, day, hour, minute
        for i in range(len(minutes_in_unit)):
            if abs(minutes) > minutes_in_unit[i]:
                return i, abs(minutes) // minutes_in_unit[i]
        return 4, 0

    def convert_prediction_to_relative_time_minutes(self, prediction):
        minutes_in_unit = [365*24*60, 30*24*60, 24*60, 60, 1] # year, month, day, hour, minute
        start_unit = np.argmax(prediction["start_unit_logits"], axis=1)
        end_unit = np.argmax(prediction["end_unit_logits"], axis=1)
        start_minutes = [minutes_in_unit[unit] * value for unit, value in zip(start_unit, prediction["start_value"])]
        start_minutes_lower = [minutes_in_unit[unit] * (value-1) for unit, value in zip(start_unit, prediction["start_value"])]
        start_minutes_upper = [minutes_in_unit[unit] * (value+1) for unit, value in zip(start_unit, prediction["start_value"])]
        end_minutes = [minutes_in_unit[unit] * value for unit, value in zip(end_unit, prediction["end_value"])]
        end_minutes_lower = [minutes_in_unit[unit] * (value-1) for unit, value in zip(end_unit, prediction["end_value"])]
        end_minutes_upper = [minutes_in_unit[unit] * (value+1) for unit, value in zip(end_unit, prediction["end_value"])]
        return (start_minutes, start_minutes_lower, start_minutes_upper), (end_minutes, end_minutes_lower, end_minutes_upper)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']  # assumes event context is in here
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        start_unit, start_value = self.convert_minutes_to_unit_value(row['start_time_minutes'] - row['admission_date_minutes'])
        end_unit, end_value = self.convert_minutes_to_unit_value(row['end_time_minutes'] - row['admission_date_minutes'])
        duration_unit, duration_value = self.convert_minutes_to_unit_value(row['duration_minutes'])

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            "start_unit": start_unit,
            "start_value": start_value,
            "end_unit": end_unit,
            "end_value": end_value,
            "duration_unit": duration_unit,
            "duration_value": duration_value,
            "start_time_minutes": row['start_time_minutes'],
            "start_upper_minutes": row['start_upper_minutes'],
            "start_lower_minutes": row['start_lower_minutes'],
            "end_time_minutes": row['end_time_minutes'],
            "end_upper_minutes": row['end_upper_minutes'],
            "end_lower_minutes": row['end_lower_minutes'],
            "duration_minutes": row['duration_minutes'],
            "duration_upper_minutes": row['duration_upper_minutes'],
            "duration_lower_minutes": row['duration_lower_minutes'],
            "admission_date_minutes": row['admission_date_minutes'],
        }
