import numpy as np
from numpy.ma.core import argmax
from torch.utils.data import Dataset
import torch

class TimelineDataset(Dataset):
    def __init__(self, dataframe, use_qa_format=False):
        self.df = dataframe
        self.use_qa_format = use_qa_format

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
        start_unit = np.argmax(prediction["start_unit_logits"].detach().cpu(), axis=1)
        end_unit = np.argmax(prediction["end_unit_logits"].detach().cpu(), axis=1)
        start_minutes = [minutes_in_unit[unit] * value for unit, value in zip(start_unit, prediction["start_value"].detach().cpu())]
        start_minutes_lower = [minutes_in_unit[unit] * (value-1) for unit, value in zip(start_unit, prediction["start_value"].detach().cpu())]
        start_minutes_upper = [minutes_in_unit[unit] * (value+1) for unit, value in zip(start_unit, prediction["start_value"].detach().cpu())]
        end_minutes = [minutes_in_unit[unit] * value for unit, value in zip(end_unit, prediction["end_value"].detach().cpu())]
        end_minutes_lower = [minutes_in_unit[unit] * (value-1) for unit, value in zip(end_unit, prediction["end_value"].detach().cpu())]
        end_minutes_upper = [minutes_in_unit[unit] * (value+1) for unit, value in zip(end_unit, prediction["end_value"].detach().cpu())]
        return (start_minutes, start_minutes_lower, start_minutes_upper), (end_minutes, end_minutes_lower, end_minutes_upper)

    def convert_time_in_minutes_to_description(self, time_in_minutes):
        time_units_to_minutes = {
            "minute": 1,
            "hour": 60,
            "day": 1440,
            "month": 43200,  # Approximating 1 month as 30 days
            "year": 1440 * 365  # Approximating 1 year as 365 days
        }
        before = False
        if time_in_minutes < 0:
            before = True
            time_in_minutes = abs(time_in_minutes)

        if time_in_minutes > time_units_to_minutes["year"]:
            unit = "year"
            value = time_in_minutes // time_units_to_minutes['year']
        elif time_in_minutes > time_units_to_minutes["month"]:
            unit = "month"
            value = time_in_minutes // time_units_to_minutes['month']
        elif time_in_minutes > time_units_to_minutes["day"]:
            unit = "day"
            value = time_in_minutes // time_units_to_minutes['day']
        elif time_in_minutes > time_units_to_minutes["hour"]:
            unit = "hour"
            value = time_in_minutes // time_units_to_minutes['hour']
        else:
            unit = "minute"
            value = time_in_minutes
        return f"{value} {unit}" + ("s" if value > 1 else "") + " " + ("before" if before else "after") + " admission"

    def __getitem_qa_format(self, idx):
        row = self.df.iloc[idx]
        question = ("You are given a patient discharge summary. An event of interest is marked within `<event>` tags in the text. "
                    "Your task is to estimate when this event started and ended relative to the patient's admission date.)\n"
                    "Provide your estimates **as clearly formatted time intervals** (e.g., \"START: 2 days before admission, " 
                    "END: 3 hours after admission\"). If exact timing is unclear, provide your best guess based on the available information.\n")
                    # "**Text:**\n"
                    # f"{row['text']}\n")
        answer = "START: " + self.convert_time_in_minutes_to_description(row["start_time_minutes"]) + ", END: " + self.convert_time_in_minutes_to_description(row["end_time_minutes"])
        return {
            "output": answer,
            "input": row['text'],
            "instruction": question,
            "row": row.to_dict()
        }

    def __getitem__(self, idx):
        if self.use_qa_format:
            return  self.__getitem_qa_format(idx)
        row = self.df.iloc[idx]
        text = row['text']  # assumes event context is in here

        start_unit, start_value = self.convert_minutes_to_unit_value(row['start_time_minutes'] - row['admission_date_minutes'])
        end_unit, end_value = self.convert_minutes_to_unit_value(row['end_time_minutes'] - row['admission_date_minutes'])
        duration_unit, duration_value = self.convert_minutes_to_unit_value(row['duration_minutes'])

        return {
            # 'input_ids': inputs['input_ids'].squeeze(0),
            # 'attention_mask': inputs['attention_mask'].squeeze(0),
            "text": text,
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
            "start_char": row['start_char'],
            "end_char": row['end_char'],
            # "start_token": start_token,
            # "end_token": end_token,
        }
