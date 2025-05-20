import os
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import torch
from dateutil.relativedelta import relativedelta


def parse_xml_file(xml_path, document_id):
    # Read and sanitize XML content
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    # Replace bare '&' not part of an entity
    xml_content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', 'n', xml_content)

    # Parse the cleaned XML
    root = ET.fromstring(xml_content)

    summary_id = Path(xml_path).stem
    events = []
    text = root.find("TEXT").text
    for elem in root.iter('EVENT'):
        event_id = elem.attrib.get('id')
        start = int(elem.attrib.get('start'))
        end = int(elem.attrib.get('end'))

        events.append({
            'summary_id': summary_id,
            'event_id': event_id,
            'start_char': start,
            'end_char': end,
            'text': text,
            'document_id': document_id
        })

    return events

def parse_tsv_file(tsv_path):
    columns = [
        'event_id', 'start_time', 'start_lower', 'start_upper',
        'duration', 'duration_lower', 'duration_upper',
        'end_time', 'end_lower', 'end_upper'
    ]
    return pd.read_csv(tsv_path, sep='\t', header=None, names=columns, index_col=False)

def find_matching_tsv(tsv_folder, summary_id):
    """
    Finds any TSV file that ends with _{summary_id}.tsv (e.g., A1_33.tsv for summary 33).
    Returns the path to the first match found.
    """
    pattern = re.compile(rf'(^.+_|){summary_id}\.tsv$')
    for fname in os.listdir(tsv_folder):
        if pattern.match(fname):
            return os.path.join(tsv_folder, fname)
    return None

def build_event_dataframe(xml_folder, tsv_folder):
    all_events = []

    for filename in os.listdir(xml_folder):
        if not filename.endswith('.xml'):
            continue

        summary_id = os.path.splitext(filename)[0]
        xml_path = os.path.join(xml_folder, filename)

        tsv_path = find_matching_tsv(tsv_folder, summary_id)
        if not tsv_path or not os.path.exists(tsv_path):
            continue

        try:
            events = parse_xml_file(xml_path, summary_id)
            if not events:
                continue

            time_df = parse_tsv_file(tsv_path)
            event_df = pd.DataFrame(events)
            merged = event_df.merge(time_df, how='inner', on='event_id')
            all_events.append(merged)
        except Exception as e:
            print(f"Error processing summary {summary_id}: {e}")

    if not all_events:
        return pd.DataFrame()

    return pd.concat(all_events, ignore_index=True)

# Helper function to parse duration string like '0Y9M0D0H0m' into timedelta
def parse_duration(duration_str):
    pattern = r"(?P<years>\d+)Y(?P<months>\d+)M(?P<days>\d+)D(?P<hours>\d+)H(?P<minutes>\d+)m"
    match = re.match(pattern, duration_str)
    if not match:
        return None
    parts = {k: int(v) for k, v in match.groupdict().items()}
    # Assume 30 days in a month and 365 days in a year for approximation
    return timedelta(
        days=parts['years'] * 365 + parts['months'] * 30 + parts['days'],
        hours=parts['hours'],
        minutes=parts['minutes']
    )

# Helper to convert timedelta back to string

def format_duration(td):
    # Ensure the timedelta is at least 1 minute
    if td.total_seconds() < 60:
        td = timedelta(minutes=1)

    total_minutes = int(td.total_seconds() // 60)
    years, rem_days = divmod(total_minutes, 365 * 24 * 60)
    months, rem_days = divmod(rem_days, 30 * 24 * 60)
    days, rem_minutes = divmod(rem_days, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    return f"{years}Y{months}M{days}D{hours}H{minutes}m"


def compute_missing_values(df):
    df = df.copy()

    rows_to_drop = []
    for i, row in df.iterrows():
        # Validate 'start_time' and 'end_time' before casting them to datetime
        start_time = None
        end_time = None
        if pd.notna(row['start_time']) and isinstance(row['start_time'], str):
            try:
                start_time = pd.to_datetime(row['start_time'])
            except Exception:
                start_time = None

        if pd.notna(row['end_time']) and isinstance(row['end_time'], str):
            try:
                end_time = pd.to_datetime(row['end_time'])
            except Exception:
                end_time = None

        # Parse 'duration' using parse_duration if it's not empty or NaN
        duration = parse_duration(row['duration']) if pd.notna(row['duration']) and row['duration'] != '' else None

        # TODO you can experiment with different functions
        def combine_distance(a, b):
            return max(a,b)

        # Compute missing value and corresponding bounds
        if start_time and end_time and not duration:
            duration = end_time - start_time
            df.at[i, 'duration'] = format_duration(duration)

            if pd.notna(row['start_lower']) and pd.notna(row['end_lower']):
                lower = combine_distance(start_time - pd.to_datetime(row['start_lower']),
                                     pd.to_datetime(row['end_lower']) - end_time)
                df.at[i, 'duration_lower'] = format_duration(lower)

            if pd.notna(row['start_upper']) and pd.notna(row['end_upper']):
                upper = combine_distance(pd.to_datetime(row['start_upper']) - start_time,
                                     end_time - pd.to_datetime(row['end_upper']))
                df.at[i, 'duration_upper'] = format_duration(upper)

        elif start_time and duration and not end_time:
            end_time = start_time + duration
            df.at[i, 'end_time'] = end_time.strftime("%Y-%m-%d %H:%M")

            if pd.notna(row['start_lower']) and pd.notna(row['duration_lower']):
                lower = combine_distance(start_time - pd.to_datetime(row['start_lower']),
                                         parse_duration(row['duration']) - parse_duration(row['duration_lower']))
                df.at[i, 'end_lower'] = (end_time - lower).strftime("%Y-%m-%d %H:%M")

            if pd.notna(row['start_upper']) and pd.notna(row['duration_upper']):
                upper = combine_distance(pd.to_datetime(row['start_upper']) - start_time,
                                     parse_duration(row['duration_upper']) - parse_duration(row['duration']))
                df.at[i, 'end_upper'] = (end_time + upper).strftime("%Y-%m-%d %H:%M")

        elif end_time and duration and not start_time:
            start_time = end_time - duration
            df.at[i, 'start_time'] = start_time.strftime("%Y-%m-%d %H:%M")

            if pd.notna(row['end_lower']) and pd.notna(row['duration_lower']):
                lower = combine_distance(pd.to_datetime(row['end_lower']) - end_time,
                                         parse_duration(row['duration']) - parse_duration(row['duration_lower']))
                df.at[i, 'start_lower'] = (start_time - lower).strftime("%Y-%m-%d %H:%M")

            if pd.notna(row['end_upper']) and pd.notna(row['duration_upper']):
                upper = combine_distance(end_time - pd.to_datetime(row['end_upper']),
                                     parse_duration(row['duration_upper']) - parse_duration(row['duration']))
                df.at[i, 'start_upper'] = (start_time + upper).strftime("%Y-%m-%d %H:%M")
        else:
            # print(f"Skipping row {i} with missing values: {row}")
            rows_to_drop.append(i)

    df.drop(rows_to_drop, inplace=True)

    return df.reset_index(drop=True)


def expand_as_minutes(df):
    """
    Expands the dataframe by adding new columns with times and durations
    as minutes since 1.1.1900.
    """
    BASE_DATETIME = pd.Timestamp("1900-01-01 00:00:00")

    df = df.copy()

    # Convert bounds (start_upper, start_lower, end_upper, end_lower) to minutes since BASE_DATETIME
    for col in ['start_time', 'start_upper', 'start_lower', 'end_time', 'end_upper', 'end_lower']:
        if col in df.columns:
            # print(col)
            df[f'{col}_minutes'] = pd.to_datetime(df[col], errors='coerce').sub(
                BASE_DATETIME).dt.total_seconds() // 60

    # Convert durations to total minutes
    for col in ['duration', 'duration_lower', 'duration_upper']:
        if col in df.columns:
            df[f'{col}_minutes'] = df[col].apply(
                lambda x: parse_duration(x).total_seconds() // 60 if pd.notna(x) and isinstance(x, str) else None)

    return df


def extract_admission_discharge_dates(df):
    """
    Extracts Admission Date and Discharge Date from text in the dataframe
    and calculates their equivalent in minutes since 1900-01-01 00:00:00.
    """
    BASE_DATETIME = pd.Timestamp("1900-01-01 00:00:00")

    # Define regex patterns for admission and discharge dates
    admission_pattern = r"(?i)Admission\s*Date\s*:\s*\n?(?P<admission_date>(\d{1,4}[/-]\d{1,2}[/-]\d{2,4}|\d{8}))"
    discharge_pattern = r"(?i)Discharge\s*Date\s*:\s*\n?(?P<discharge_date>(\d{1,4}[/-]\d{1,2}[/-]\d{2,4}|\d{8}))"

    # Initialize new columns
    df['admission_date'] = None
    df['discharge_date'] = None

    # Extract dates from the text column
    for i, row in df.iterrows():
        if 'text' in df.columns and pd.notna(row['text']):
            admission_match = re.search(admission_pattern, row['text'])
            discharge_match = re.search(discharge_pattern, row['text'])

            # Extract the matched dates
            if admission_match:
                time = pd.to_datetime(admission_match.group('admission_date'), format='%m-%d-%y', errors='coerce') if '-' in admission_match.group('admission_date') else pd.to_datetime(admission_match.group('admission_date'), format='%Y%m%d', errors='coerce')
                if not pd.notna(time):
                    time = pd.to_datetime(admission_match.group('admission_date'), errors='coerce')
                df.at[i, 'admission_date'] = time

            if discharge_match:
                time = pd.to_datetime(discharge_match.group('discharge_date'), format='%m-%d-%y', errors='coerce') if '-' in discharge_match.group('discharge_date') else pd.to_datetime(discharge_match.group('discharge_date'), format='%Y%m%d', errors='coerce')
                if not pd.notna(time):
                    time = pd.to_datetime(discharge_match.group('discharge_date'), errors='coerce')
                df.at[i, 'discharge_date'] = time

                # Convert the dates to minutes since BASE_DATETIME
    df['admission_date_minutes'] = df['admission_date'].map(
        lambda x: (x - BASE_DATETIME).total_seconds() // 60 if pd.notna(x) else None)
    df['discharge_date_minutes'] = df['discharge_date'].map(
        lambda x: (x - BASE_DATETIME).total_seconds() // 60 if pd.notna(x) else None)

    return df


def load_i2b2_absolute_data(test_split=False):
    # Example usage
    if test_split:
        xml_folder = 'data/i2b2-test'
    else:
        xml_folder = 'data/i2b2'
    tsv_folder = 'data/i2b2-absolute'
    df = build_event_dataframe(xml_folder, tsv_folder)

    df = compute_missing_values(df)
    df = expand_as_minutes(df)
    df = extract_admission_discharge_dates(df)
    return df


if __name__ == '__main__':
    df = load_i2b2_absolute_data(test_split=False)
    df = df.iloc[:10]
    torch.save(df, 'demo_data.pt')
    print(df.columns)