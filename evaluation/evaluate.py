import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import evaluate_temporal_predictions, compute_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else device

# def evaluate_document(document_id: str, predictions: list[dict], labels: dict[str, dict], combined_scores=None):
#     if combined_scores is None:
#         combined_scores = {
#             "correct_start": 0,
#             "correct_end": 0,
#             "n": 0,
#         }
#         document_correct_start = 0
#         document_correct_end = 0
#         document_n = 0
#         for prediction in predictions:
#             event_id = prediction["event_id"]
#             label_start = [labels[event_id][value] for value in ["start_time_minutes", "start_lower_minutes", "start_upper_minutes"]]
#             label_end = [labels[event_id][value] for value in ["end_time_minutes", "end_lower_minutes", "end_upper_minutes"]]
#             predicted_start = prediction["start_time_minutes"]
#             predicted_end = prediction["end_time_minutes"]
#             if label_start[1] <= predicted_start <= label_start[2]:
#                 document_correct_start += 1
#                 combined_scores["correct_start"] += 1
#             if label_end[1] <= predicted_end <= label_end[2]:
#                 document_correct_end += 1
#                 combined_scores["correct_end"] += 1
#             document_n += 1
#             combined_scores["n"] += 1
#             
#         precision_start = document_correct_start / document_n
#         precision_end = document_correct_end / document_n
#         print(f"Document: {document_id}")
#         print(f"Precision (Start): {precision_start}")
#         print(f"Precision (End): {precision_end}")
# 
#         print("Updated combined scores:")
#         precision_start = combined_scores["correct_start"] / combined_scores["n"]
#         precision_end = combined_scores["correct_end"] / combined_scores["n"]
#         print(f"Precision (Start): {precision_start}")
#         print(f"Precision (End): {precision_end}")
#         print()
#         return combined_scores
# 
# 
# def evaluate_model_batch(df, model):
#     true_labels = []
#     pred_labels = []
#     combined_scores = None
#     for document_id, document_results in model.predict(df):
#         labels = build_ground_truth(df, document_id)
#         combined_scores = evaluate_document(document_id, document_results, labels, combined_scores=combined_scores)
#         pass
#     pass
#     return {
#         # TODO compute the main scores
#         "precision": combined_scores["correct_start"] / combined_scores["n"],
#     }
#     pass
# 
# 
# def evaluate_model(df, model):
#     true_labels = []
#     pred_labels = []
# 
#     for _, row in df.iterrows():
#         # Get model prediction
#         prediction = model.predict(row)
#         pred_start, pred_end = prediction['start_minutes'], prediction['end_minutes']
# 
#         # Check if predicted start and end times fall within the gold standard range
#         start_correct = row['start_lower_minutes'] <= pred_start <= row['start_upper_minutes']
#         end_correct = row['end_lower_minutes'] <= pred_end <= row['end_upper_minutes']
# 
#         print("predicted start:", pred_start)
#         print("predicted end:", pred_end)
#         print("gold start:", row['start_lower_minutes'], "-", row['start_upper_minutes'])
#         print("gold end:", row['end_lower_minutes'], "-", row['end_upper_minutes'])
#         print("start correct:", start_correct)
#         print("end correct:", end_correct)
#         print("=" * 30)
#         print(
#             f"Predicted start: {pred_start} ({'correct' if start_correct else 'incorrect'}), "
#             f"Predicted end: {pred_end} ({'correct' if end_correct else 'incorrect'})"
#         )
# 
#         # Label is 1 if both start and end are within bounds, 0 otherwise
#         true_labels.append(1)
#         pred_labels.append(1 if start_correct and end_correct else 0)
# 
#     # Compute metrics
#     precision = precision_score(true_labels, pred_labels, zero_division=0)
#     recall = recall_score(true_labels, pred_labels, zero_division=0)
#     f1 = f1_score(true_labels, pred_labels, zero_division=0)
# 
#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1
#     }
# 
# 
# def build_ground_truth(df, document_id):
#     events = {}
#     for i, row in df.iterrows():
#         if row['document_id'] != document_id:
#             continue
#         event_id = row['event_id']
#         start_time_minutes = row['start_time_minutes']
#         start_upper_minutes = row['start_upper_minutes']
#         start_lower_minutes = row['start_lower_minutes']
#         end_time_minutes = row['end_time_minutes']
#         end_upper_minutes = row['end_upper_minutes']
#         end_lower_minutes = row['end_lower_minutes']
#         duration_minutes = row['duration_minutes']
#         duration_lower_minutes = row['duration_lower_minutes']
#         duration_upper_minutes = row['duration_upper_minutes']
#         admission_date_minutes = row['admission_date_minutes']
# 
#         events[event_id] = {
#             "start_time_minutes": start_time_minutes,
#             "start_upper_minutes": start_upper_minutes,
#             "start_lower_minutes": start_lower_minutes,
#             "end_time_minutes": end_time_minutes,
#             "end_upper_minutes": end_upper_minutes,
#             "end_lower_minutes": end_lower_minutes,
#             "duration_minutes": duration_minutes,
#             "duration_lower_minutes": duration_lower_minutes,
#             "duration_upper_minutes": duration_upper_minutes,
#             "admission_date_minutes": admission_date_minutes
#         }
#     return events
# 
# 
# def evaluate_models():
#     api_key = os.getenv("GEMINI_API_KEY")
#     # predictor = ZeroShotPromptingModel(local_llm.OllamaModel(model_name="gemma3:27b"), use_structured_response=False)
#     # predictor = ZeroShotPromptingModel(gemini.GeminiModel(api_key))
#     predictor = EventTimePredictor(gemini.GeminiModel(api_key), use_structured_response=False)
# 
#     df = load_i2b2_absolute_data()
#     combined_scores = None
#     for document_id, document_results in evaluate_model_batch(df, predictor):
#         labels = build_ground_truth(df, document_id)
#         combined_scores = evaluate_document(document_id, document_results, labels, combined_scores=combined_scores)

BASE_DATETIME = datetime.datetime(1900, 1, 1, 0, 0, 0)
def convert_minutes_to_datetime(minutes_since_base, duration=False):
    """Converts minutes since BASE_DATETIME to a datetime object."""
    delta = datetime.timedelta(minutes=int(minutes_since_base))
    if duration:
        return delta
    else:
        return BASE_DATETIME + delta

def show_predictions(predicted_starts, predicted_ends, predicted_durations, gold_starts, gold_ends, gold_durations, batch):
    for ((pred_s, _, _), (pred_e, _, _), (pred_d, _, _), (gold_s, gold_s_lower, gold_s_upper), (gold_e, gold_e_lower, gold_e_upper), (gold_d, gold_d_lower, gold_d_upper),
         text, start_char, end_char, admission_date_minutes) in (
            zip(predicted_starts, predicted_ends, predicted_durations,gold_starts, gold_ends, gold_durations,
                batch["text"], batch["start_char"], batch["end_char"], batch["admission_date_minutes"])):
        print(text[:start_char] + "<e>" + text[start_char:end_char] + "</e>" + text[end_char:])
        print("Admission:")
        print(convert_minutes_to_datetime(admission_date_minutes))
        print("Start:")

        print("Predicted:", convert_minutes_to_datetime(pred_s),
              "Gold:", convert_minutes_to_datetime(gold_s),
              "(" + str(convert_minutes_to_datetime(gold_s_lower)) + " - " + str(convert_minutes_to_datetime(gold_s_upper)) + ")")
        print("End:")
        print("Predicted:", convert_minutes_to_datetime(pred_e),
              "Gold:", convert_minutes_to_datetime(gold_e),
              "(" + str(convert_minutes_to_datetime(gold_e_lower)) + " - " + str(convert_minutes_to_datetime(gold_e_upper)) + ")")
        print("Duration:")
        print("Predicted:", convert_minutes_to_datetime(pred_d, duration=True),
              "Gold:", convert_minutes_to_datetime(gold_d, duration=True),
              "(" + str(convert_minutes_to_datetime(gold_d_lower, duration=True)) + " - " + str(convert_minutes_to_datetime(gold_d_upper, duration=True)) + ")")
    pass

def evaluate(model, dataloader, simplified_model) -> (float, dict):
    model.eval()
    total_eval_loss = 0
    predicted_starts = []
    predicted_ends = []
    predicted_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []

    with torch.no_grad():
        for batch in dataloader:
            if simplified_model:
                labels = (
                    batch['start_time_minutes'] - batch['admission_date_minutes'],
                    batch['end_time_minutes'] - batch['admission_date_minutes'],
                    batch['duration_minutes']
                )
            else:
                labels = (
                    batch['start_unit'].to(device),
                    batch['start_value'].to(device),
                    batch['end_unit'].to(device),
                    batch['end_value'].to(device),
                    batch['duration_unit'].to(device),
                    batch['duration_value'].to(device)
                )

            admission_times = batch['admission_date_minutes']
            if simplified_model:
                outputs = model(
                    text=batch['text'],
                    labels=labels,
                    start_char_index=batch['start_char'],
                    end_char_index=batch['end_char']
                )
                total_eval_loss += outputs['loss'].item()
                start_minutes = (outputs['predictions'][:, 0].cpu() + admission_times).tolist()
                end_minutes = (outputs['predictions'][:, 1].cpu() + admission_times).tolist()
                duration_minutes = (outputs['predictions'][:, 2].cpu() + admission_times).tolist()
                start_minutes = [(t, t - 60, t+60) for t in start_minutes] # TODO implement better lower and upper bound guess
                end_minutes = [(t, t - 60, t+60) for t in end_minutes]
                duration_minutes = [(t, t - 60, t+60) for t in duration_minutes]
            else:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=labels
                )
                start_minutes, end_minutes = dataloader.dataset.convert_prediction_to_relative_time_minutes(outputs)
                start_minutes = [(float(t+at),float(l+at),float(u+at)) for (t,l,u), at in zip(zip(*start_minutes), admission_times)]
                end_minutes = [(float(t+at),float(l+at),float(u+at)) for (t,l,u), at in zip(zip(*end_minutes), admission_times)]

            predicted_starts += start_minutes
            predicted_ends += end_minutes
            predicted_durations += duration_minutes
            gold_starts += [(float(t),float(l),float(u)) for t,l,u in zip(batch["start_time_minutes"], batch["start_lower_minutes"], batch["start_upper_minutes"])]
            gold_ends += [(float(t),float(l),float(u)) for t,l,u in zip(batch["end_time_minutes"], batch["end_lower_minutes"], batch["end_upper_minutes"])]
            gold_durations += [(float(t),float(l),float(u)) for t,l,u in zip(batch["duration_minutes"], batch["duration_lower_minutes"], batch["duration_upper_minutes"])]
            eval_metrics = compute_metrics(predicted_starts, predicted_ends, predicted_durations, gold_starts, gold_ends, gold_durations)
            show_predictions(predicted_starts, predicted_ends, predicted_durations, gold_starts, gold_ends, gold_durations, batch)
    print("End of epoch results on eval:")

    print(eval_metrics)

    avg_eval_loss = total_eval_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    return avg_eval_loss, eval_metrics

if __name__ == '__main__':
    from train.train_bert_model import load_model
    model, tokenizer = load_model("results/testni_model")
    dataframe = load_i2b2_absolute_data()
    dataframe_test = load_i2b2_absolute_data(test_split=True)
    dataset = TimelineDataset(dataframe)
    dataset_test = TimelineDataset(dataframe_test)
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    evaluate(model, dataloader, dataloader_test)