import dspy
import torch
from datetime import datetime

from data_loaders.i2b2dataLoader import load_absolute_data
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data


def preprocess(row, log_file='preprocess.log'):
    """
    Preprocess a DataFrame row by updating the 'text' field using the Gemma 3 27b model.

    Parameters:
        row (pd.Series): A single row from a DataFrame.

    Returns:
        pd.Series: The updated row with paraphrased 'text'.
    """
    lm = dspy.LM('ollama_chat/gemma3:27b', api_base='http://localhost:11434', api_key='')

    # Extract the required fields from the row
    text = row.get('text', "")  # Original text
    event = text[row['start_char']:row['end_char']]
    text_with_markers = text[:row['start_char']] + f"<event>{event}</event>" + text[row['end_char']:]

    # Prepare the paraphrasing prompt
    prompt = f"""Paraphrase the following discharge summary to make it clear when the <event>{event}</event> occurred \
with regards to the admission date. In the text, mark the {event} with the <event> marker.

{text_with_markers}
"""
    response = lm(prompt)[0]
    updated_text = response.strip()



    if updated_text.find('<event>') == -1 or updated_text.find('</event>') == -1:
        print("event mention not found after preprocessing")
        return row
    new_start_char = updated_text.find('<event>') + len('<event>')
    new_end_char = updated_text.find('</event>')

    # Update the row's text field
    row['text'] = updated_text
    row['start_char'] = new_start_char
    row['end_char'] = new_end_char
    try:
        with open(log_file, 'a') as log:
            log.write(f"{datetime.now().isoformat()} - Event: {event} - Updated text: {updated_text}\n")
    except Exception as e:
        print(f"Logging failed: {e}")
    return row

if __name__ == '__main__':
    # df = load_data()
    # df = load_data_event_extraction()
    df = load_i2b2_absolute_data(test_split=False)
    df = df.iloc[:10]
    df = df.apply(lambda row: preprocess(row, log_file='preprocess.log'), axis=1)
    torch.save(df, "data/i2b2_train_absolute_preprocessed.pt")

    df = load_i2b2_absolute_data(test_split=True)

    df = df.apply(lambda row: preprocess(row, log_file='preprocess.log'), axis=1)
    torch.save(df, "data/i2b2_test_absolute_preprocessed.pt")
    # df = load_data_temporal_attributes()
    pass