import json

import numpy as np
import pandas as pd
from numpy.ma.core import argmax
from torch.utils.data import Dataset
import torch
from datetime import datetime, timedelta


class SelectingTimeExressionsDataset(Dataset):
    def __init__(self, dataframe):
        self.df = self.construct_expanded_df(dataframe)

    def construct_expanded_df(self, df):
        expanded_rows = []
        for _, row in df.iterrows():
            text = row['text']
            start_char = row['start_char']
            end_char = row['end_char']
            temporal_expressions = row['temporal_expressions']
            # find the closest expression to the mean of row['start_time_minutes'] and row['end_time_minutes'] with regards to expression["value_minutes"]
            mean_time = (row['start_time_minutes'] + row['end_time_minutes']) / 2

            # add time in monutes
            for expression in temporal_expressions:
                time_value = expression["value"]
                if 'T' in time_value or 't' in time_value:
                    dt = datetime.strptime(time_value.replace('T', ' ').replace('t', ' ').replace('- ', ' '), '%Y-%m-%d %H:%M')
                else:
                    dt = (
                        datetime.strptime(time_value, '%Y-%m-%d') if len(time_value) == 10
                        else datetime.strptime(time_value, '%Y-%m') if len(time_value) == 7
                        else datetime.strptime(time_value, '%Y')
                    )
                reference_date = datetime(1900, 1, 1)
                delta = dt - reference_date
                minutes_since_reference = delta.total_seconds() // 60
                expression['value_minutes'] = minutes_since_reference

            # Find the closest expression to the mean time
            closest_expression = None
            closest_diff = float('inf')
            for expression in temporal_expressions:
                if 'value_minutes' in expression:
                    diff = abs(expression['value_minutes'] - mean_time)
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_expression = expression

            if not closest_expression:
                continue

            for expression in temporal_expressions:
                if closest_expression["time_id"] == expression["time_id"]:
                    closest = 1
                else:
                    closest = 0

                new_row = {
                    'text': text,
                    'document_id': row['document_id'],
                    'event_id': row['event_id'],
                    "start_time_minutes": row['start_time_minutes'],
                    "end_time_minutes": row['end_time_minutes'],
                    'start_char': start_char,
                    'end_char': end_char,
                    'expression_char_start': expression["start"],
                    'expression_char_end': expression["end"],
                    'value': expression["value"],
                    'value_minutes': expression["value_minutes"],
                    "admission_date_minutes": row['admission_date_minutes'],
                    "closest": closest,
                }
                expanded_rows.append(new_row)
        return pd.DataFrame(expanded_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {"row": {
            'text': row["text"],
            'document_id': row["document_id"],
            'event_id': row["event_id"],
            "start_time_minutes": row["start_time_minutes"],
            "end_time_minutes": row["end_time_minutes"],
            'start_char': row["start_char"],
            'end_char': row["end_char"],
            'expression_char_start': row["expression_char_start"],
            'expression_char_end': row["expression_char_end"],
            'value': row["value"],
            'value_minutes': row["value_minutes"],
            "admission_date_minutes": row["admission_date_minutes"],
            "closest": row["closest"]
        }}

class TimelineDataset(Dataset):
    def __init__(self, dataframe, use_qa_format=False, use_absolute_qa=True):
        self.df = dataframe
        self.use_qa_format = use_qa_format
        self.use_absolute_qa = use_absolute_qa

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

    def convert_time_in_minutes_to_iso(self, time_in_minutes):
        reference_date = datetime(1900, 1, 1)
        delta = timedelta(minutes=time_in_minutes)
        target_date = reference_date + delta
        return target_date.strftime('%Y-%m-%d %H:%M:%S')

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
        medical_text = row['text']
        # annotate event
        medical_text = (medical_text[:row['start_char']] + "<event>" + medical_text[row['start_char']:row['end_char']] +
                        "</event>" + medical_text[row['end_char']:])
        if self.use_absolute_qa:
            question = (
                "You are given a patient discharge summary. An event of interest is marked within `<event>` tags in the text. "
                "Your task is to estimate when this event started and ended.\n"
                "Provide your estimates **as clearly formatted time in the format yyyy-MM-dd HH:mm:ss** (e.g., \"START: 2025-08-11 09:00:00, "
                "END: 2025-08-11 10:30:00\"). If exact timing is unclear, provide your best guess based on the available information. "
                "When only the date is given in a document, guess the start and end times.\n")
            # "**Text:**\n"
            # f"{row['text']}\n")
            answer = "START: " + self.convert_time_in_minutes_to_iso(
                row["start_time_minutes"]) + ", END: " + self.convert_time_in_minutes_to_iso(
                row["end_time_minutes"])
        else:
            medical_text = row['text']
            # annotate event
            medical_text = (
                        medical_text[:row['start_char']] + "<event>" + medical_text[row['start_char']:row['end_char']] +
                        "</event>" + medical_text[row['end_char']:])
            question = ("You are given a patient discharge summary. An event of interest is marked within `<event>` tags in the text. "
                        "Your task is to estimate when this event started and ended relative to the patient's admission date.\n"
                        "Also extract events if there is a clear temporal relation between two events, either the amount of time that has passed or a relation like before/after.\n"
                        "Provide your estimates **as clearly formatted time intervals** (e.g., \"START: 2 days before admission, " 
                        "END: 3 hours after admission\"). If exact timing is unclear, provide your best guess based on the available information.\n")
                        # "**Text:**\n"
                        # f"{row['text']}\n")
            answer = "START: " + self.convert_time_in_minutes_to_description(row["start_time_minutes"]) + ", END: " + self.convert_time_in_minutes_to_description(row["end_time_minutes"])
        return {
            "output": answer,
            "input": medical_text,
            "instruction": question,
            "row": row.to_dict()
        }

    def update_time_expressions(self, temporal_expressions):
        for i, expression in enumerate(temporal_expressions):
            time_value = expression["value"]
            if 'T' in time_value or 't' in time_value:
                dt = datetime.strptime(time_value.replace('T', ' ').replace('t', ' ').replace('- ', ' '),
                                       '%Y-%m-%d %H:%M')
            else:
                dt = (
                    datetime.strptime(time_value, '%Y-%m-%d') if len(time_value) == 10
                    else datetime.strptime(time_value, '%Y-%m') if len(time_value) == 7
                    else datetime.strptime(time_value, '%Y')
                )
            reference_date = datetime(1900, 1, 1)
            delta = dt - reference_date
            minutes_since_reference = delta.total_seconds() // 60

            expression['value_minutes'] = minutes_since_reference
            temporal_expressions[i] = expression
        return temporal_expressions

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
            "document_id": row['document_id'],
            "event_id": row['event_id'],
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
            "temporal_expressions": json.dumps(self.update_time_expressions(row['temporal_expressions'])),
            # "start_token": start_token,
            # "end_token": end_token,
        }
