from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from llm_testing.llm_models import local_llm, gemini
from llm_testing.absolute_time_predictor import ZeroShotPromptingModel
from sklearn.metrics import precision_score, recall_score, f1_score

import os

from llm_testing.multiple_times_predictor import EventTimePredictor

def evaluate_document(document_id: str, predictions: list[dict], labels: dict[str, dict], combined_scores=None):
    if combined_scores is None:
        combined_scores = {
            "correct_start": 0,
            "correct_end": 0,
            "n": 0,
        }
        document_correct_start = 0
        document_correct_end = 0
        document_n = 0
        for prediction in predictions:
            event_id = prediction["event_id"]
            label_start = [labels[event_id][value] for value in ["start_time_minutes", "start_lower_minutes", "start_upper_minutes"]]
            label_end = [labels[event_id][value] for value in ["end_time_minutes", "end_lower_minutes", "end_upper_minutes"]]
            predicted_start = prediction["start_time_minutes"]
            predicted_end = prediction["end_time_minutes"]
            if label_start[1] <= predicted_start <= label_start[2]:
                document_correct_start += 1
                combined_scores["correct_start"] += 1
            if label_end[1] <= predicted_end <= label_end[2]:
                document_correct_end += 1
                combined_scores["correct_end"] += 1
            document_n += 1
            combined_scores["n"] += 1
            
        precision_start = document_correct_start / document_n
        precision_end = document_correct_end / document_n
        print(f"Document: {document_id}")
        print(f"Precision (Start): {precision_start}")
        print(f"Precision (End): {precision_end}")

        print("Updated combined scores:")
        precision_start = combined_scores["correct_start"] / combined_scores["n"]
        precision_end = combined_scores["correct_end"] / combined_scores["n"]
        print(f"Precision (Start): {precision_start}")
        print(f"Precision (End): {precision_end}")
        print()
        return combined_scores


def evaluate_model_batch(df, model):
    true_labels = []
    pred_labels = []
    combined_scores = None
    for document_id, document_results in model.predict(df):
        labels = build_ground_truth(df, document_id)
        combined_scores = evaluate_document(document_id, document_results, labels, combined_scores=combined_scores)
        pass
    pass
    return {
        # TODO compute the main scores
        "precision": combined_scores["correct_start"] / combined_scores["n"],
    }
    pass


def evaluate_model(df, model):
    true_labels = []
    pred_labels = []

    for _, row in df.iterrows():
        # Get model prediction
        prediction = model.predict(row)
        pred_start, pred_end = prediction['start_minutes'], prediction['end_minutes']

        # Check if predicted start and end times fall within the gold standard range
        start_correct = row['start_lower_minutes'] <= pred_start <= row['start_upper_minutes']
        end_correct = row['end_lower_minutes'] <= pred_end <= row['end_upper_minutes']

        print("predicted start:", pred_start)
        print("predicted end:", pred_end)
        print("gold start:", row['start_lower_minutes'], "-", row['start_upper_minutes'])
        print("gold end:", row['end_lower_minutes'], "-", row['end_upper_minutes'])
        print("start correct:", start_correct)
        print("end correct:", end_correct)
        print("=" * 30)
        print(
            f"Predicted start: {pred_start} ({'correct' if start_correct else 'incorrect'}), "
            f"Predicted end: {pred_end} ({'correct' if end_correct else 'incorrect'})"
        )

        # Label is 1 if both start and end are within bounds, 0 otherwise
        true_labels.append(1)
        pred_labels.append(1 if start_correct and end_correct else 0)

    # Compute metrics
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def build_ground_truth(df, document_id):
    events = {}
    for i, row in df.iterrows():
        if row['document_id'] != document_id:
            continue
        event_id = row['event_id']
        start_time_minutes = row['start_time_minutes']
        start_upper_minutes = row['start_upper_minutes']
        start_lower_minutes = row['start_lower_minutes']
        end_time_minutes = row['end_time_minutes']
        end_upper_minutes = row['end_upper_minutes']
        end_lower_minutes = row['end_lower_minutes']
        duration_minutes = row['duration_minutes']
        duration_lower_minutes = row['duration_lower_minutes']
        duration_upper_minutes = row['duration_upper_minutes']
        admission_date_minutes = row['admission_date_minutes']

        events[event_id] = {
            "start_time_minutes": start_time_minutes,
            "start_upper_minutes": start_upper_minutes,
            "start_lower_minutes": start_lower_minutes,
            "end_time_minutes": end_time_minutes,
            "end_upper_minutes": end_upper_minutes,
            "end_lower_minutes": end_lower_minutes,
            "duration_minutes": duration_minutes,
            "duration_lower_minutes": duration_lower_minutes,
            "duration_upper_minutes": duration_upper_minutes,
            "admission_date_minutes": admission_date_minutes
        }
    return events


def evaluate_models():
    api_key = os.getenv("GEMINI_API_KEY")
    # predictor = ZeroShotPromptingModel(local_llm.OllamaModel(model_name="gemma3:27b"), use_structured_response=False)
    # predictor = ZeroShotPromptingModel(gemini.GeminiModel(api_key))
    predictor = EventTimePredictor(gemini.GeminiModel(api_key), use_structured_response=False)

    df = load_i2b2_absolute_data()
    combined_scores = None
    for document_id, document_results in evaluate_model_batch(df, predictor):
        labels = build_ground_truth(df, document_id)
        combined_scores = evaluate_document(document_id, document_results, labels, combined_scores=combined_scores)

if __name__ == '__main__':
    evaluate_models()