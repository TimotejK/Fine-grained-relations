import math
from math import floor

import torch
from sklearn.metrics import precision_score, recall_score, f1_score

documents_results = {}
def store_prediction_for_error_analysis(model_id, document_id, text, event_id, event_char_start, event_char_end, predicted_start, predicted_end, predicted_duration, gold_start, gold_end, gold_duration):
    global documents_results
    if document_id not in documents_results:
        documents_results[document_id] = {"events": [], "text": text}
    event_results = {
        "event_id": event_id,
        "event_char_start": event_char_start,
        "event_char_end": event_char_end,
        "pred_start": predicted_start,
        "pred_end": predicted_end,
        "pred_duration": predicted_duration,
        "gold_start": gold_start,
        "gold_end": gold_end,
        "gold_duration": gold_duration,
    }
    documents_results[document_id]["events"].append(event_results)
    torch.save(documents_results, f"error_analysis_{model_id}.pt")


def compute_metrics(predicted_starts, predicted_ends, predicted_durations, gold_starts, gold_ends, gold_durations):
    assert len(predicted_starts) == len(predicted_ends) == len(gold_starts) == len(gold_ends), "All lists must be of equal length."
    correct_start = 0
    correct_end = 0
    correct_duration = 0
    correct_start_expanded = 0
    correct_end_expanded = 0
    correct_duration_expanded = 0
    n = 0

    # For MAE calculation
    start_errors = []
    end_errors = []
    duration_errors = []
    for i in range(len(predicted_starts)):
        if isinstance(predicted_starts[i], tuple):
            pred_start_time = predicted_starts[i][0]
            pred_start_lower = predicted_starts[i][1]
            pred_start_upper = predicted_starts[i][2]
        else:
            pred_start_time = predicted_starts[i]
            pred_start_lower = predicted_starts[i] - 60
            pred_start_upper = predicted_starts[i] + 60
        
        if isinstance(predicted_ends[i], tuple):
            pred_end_time = predicted_ends[i][0]
            pred_end_lower = predicted_ends[i][1]
            pred_end_upper = predicted_ends[i][2]
        else:
            pred_end_time = predicted_ends[i]
            pred_end_lower = predicted_ends[i] - 60
            pred_end_upper = predicted_ends[i] + 60

        if isinstance(predicted_durations[i], tuple):
            pred_dur_time = predicted_durations[i][0]
            pred_dur_lower = predicted_durations[i][1]
            pred_dur_upper = predicted_durations[i][2]
        else:
            pred_dur_time = predicted_durations[i]
            pred_dur_lower = predicted_durations[i] - 60
            pred_dur_upper = predicted_durations[i] + 60

        # Check for NaN values and skip or handle them
        if (math.isnan(pred_start_time) or math.isnan(pred_end_time) or math.isnan(pred_dur_time)):
            print(f"Warning: NaN detected in predictions for sample {i}")
            print(f"  pred_start_time: {pred_start_time}")
            print(f"  pred_end_time: {pred_end_time}")
            print(f"  pred_dur_time: {pred_dur_time}")

        gold_start_lower = gold_starts[i][1]
        gold_start_upper = gold_starts[i][2]
        gold_end_lower = gold_ends[i][1]
        gold_end_upper = gold_ends[i][2]
        gold_duration_lower = gold_durations[i][1]
        gold_duration_upper = gold_durations[i][2]

        if gold_start_lower > gold_start_upper:
            gold_start_upper, gold_start_lower = gold_start_lower, gold_start_upper
        if gold_end_lower > gold_end_upper:
            gold_end_upper, gold_end_lower = gold_end_lower, gold_end_upper
        if gold_duration_lower > gold_duration_upper:
            gold_duration_upper, gold_duration_lower = gold_duration_lower, gold_duration_upper


        import pandas as pd
        import datetime

        # Safe timestamp conversion with NaN checking
        try:
            # print("predicted start:", pd.Timestamp("1900-01-01 00:00:00") + datetime.timedelta(minutes=pred_start_time))
            # print("true start (lower):", pd.Timestamp("1900-01-01 00:00:00") + datetime.timedelta(minutes=gold_start_lower))
            # print("true start (upper):", pd.Timestamp("1900-01-01 00:00:00") + datetime.timedelta(minutes=gold_start_upper))
            # print("predicted end:", pd.Timestamp("1900-01-01 00:00:00") + datetime.timedelta(minutes=pred_end_time))
            # print("true end (lower):", pd.Timestamp("1900-01-01 00:00:00") + datetime.timedelta(minutes=gold_end_lower))
            # print("true end (upper):", pd.Timestamp("1900-01-01 00:00:00") + datetime.timedelta(minutes=gold_end_upper))
            # print()
            pass
        except (ValueError, OverflowError) as e:
            print(f"Error in timestamp conversion for sample {i}: {e}")
            print(f"Values: start={pred_start_time}, end={pred_end_time}, duration={pred_dur_time}")
            continue

        def round_to_day(minutes):
            if math.isnan(minutes):
                return float('nan')
            return floor(minutes / (24 * 60)) * (24 * 60)

        if gold_start_lower <= pred_start_time <= gold_start_upper:
            correct_start += 1
        if gold_end_lower <= pred_end_time <= gold_end_upper:
            correct_end += 1

        if gold_duration_lower <= pred_dur_time <= gold_duration_upper:
            correct_duration += 1

        if round_to_day(gold_start_lower) <= round_to_day(pred_start_time) <= round_to_day(gold_start_upper):
            correct_start_expanded += 1
        if round_to_day(gold_end_lower) <= round_to_day(pred_end_time) <= round_to_day(gold_end_upper):
            correct_end_expanded += 1

        if round_to_day(gold_duration_lower) <= round_to_day(pred_dur_time) <= round_to_day(gold_duration_upper):
            correct_duration_expanded += 1

        # Calculate MAE using the middle of the gold range as the target
        gold_start_mid = gold_starts[i][0] if isinstance(gold_starts[i], tuple) else gold_starts[i]
        gold_end_mid = gold_ends[i][0] if isinstance(gold_ends[i], tuple) else gold_ends[i]
        gold_duration_mid = gold_durations[i][0] if isinstance(gold_durations[i], tuple) else gold_durations[i]

        if not math.isnan(pred_start_time) and not math.isnan(gold_start_mid):
            start_errors.append(abs(pred_start_time - gold_start_mid))
        if not math.isnan(pred_end_time) and not math.isnan(gold_end_mid):
            end_errors.append(abs(pred_end_time - gold_end_mid))
        if not math.isnan(pred_dur_time) and not math.isnan(gold_duration_mid):
            duration_errors.append(abs(pred_dur_time - gold_duration_mid))

        n += 1

    # Calculate MAE
    start_mae = sum(start_errors) / len(start_errors) if len(start_errors) > 0 else float('inf')
    end_mae = sum(end_errors) / len(end_errors) if len(end_errors) > 0 else float('inf')
    duration_mae = sum(duration_errors) / len(duration_errors) if len(duration_errors) > 0 else float('inf')

    return {
        "correct_start": correct_start,
        "correct_end": correct_end,
        "correct_duration": correct_duration,
        "correct_start_expanded": correct_start_expanded,
        "correct_end_expanded": correct_end_expanded,
        "correct_duration_expanded": correct_duration_expanded,
        "n": n,
        "precision_start": correct_start / n if n > 0 else 0,
        "precision_end": correct_end / n if n > 0 else 0,
        "precision_duration": correct_duration / n if n > 0 else 0,
        "precision_start_expanded": correct_start_expanded / n if n > 0 else 0,
        "precision_end_expanded": correct_end_expanded / n if n > 0 else 0,
        "precision_duration_expanded": correct_duration_expanded / n if n > 0 else 0,
        # MAE metrics
        "start_mae": start_mae,
        "end_mae": end_mae,
        "duration_mae": duration_mae,
        # Accuracy metrics (using precision as accuracy)
        "start_accuracy": correct_start / n if n > 0 else 0,
        "end_accuracy": correct_end / n if n > 0 else 0,
        "duration_accuracy": correct_duration / n if n > 0 else 0,
    }
            

def evaluate_temporal_predictions(predicted_start, predicted_end, gold_start, gold_end):
    assert len(predicted_start) == len(predicted_end) == len(gold_start) == len(gold_end), "All lists must be of equal length."

    start_preds, start_truths = [], []
    end_preds, end_truths = [], []
    pair_preds, pair_truths = [], []

    overlap_precisions, overlap_recalls = [], []

    for i in range(len(predicted_start)):
        # Extract values
        pred_start, _, _ = predicted_start[i]
        gold_start_val, gold_start_lb, gold_start_ub = gold_start[i]

        pred_end, _, _ = predicted_end[i]
        gold_end_val, gold_end_lb, gold_end_ub = gold_end[i]

        # Gold truth: assume 1 for all
        start_truths.append(1)
        end_truths.append(1)
        pair_truths.append(1)

        # Normalization correctness
        start_correct = gold_start_lb <= pred_start <= gold_start_ub
        end_correct = gold_end_lb <= pred_end <= gold_end_ub

        start_preds.append(1 if start_correct else 0)
        end_preds.append(1 if end_correct else 0)
        pair_preds.append(1 if start_correct and end_correct else 0)

        # --- Overlap calculation ---
        pred_span_start = min(pred_start, pred_end)
        pred_span_end = max(pred_start, pred_end)

        gold_span_start = min(gold_start_val, gold_end_val)
        gold_span_end = max(gold_start_val, gold_end_val)

        intersection = max(0, min(pred_span_end, gold_span_end) - max(pred_span_start, gold_span_start))
        pred_duration = max(1e-5, pred_span_end - pred_span_start)
        gold_duration = max(1e-5, gold_span_end - gold_span_start)

        overlap_precision = intersection / pred_duration
        overlap_recall = intersection / gold_duration

        overlap_precisions.append(overlap_precision)
        overlap_recalls.append(overlap_recall)

    # Mean overlap-based metrics
    avg_overlap_precision = sum(overlap_precisions) / len(overlap_precisions)
    avg_overlap_recall = sum(overlap_recalls) / len(overlap_recalls)
    if avg_overlap_precision + avg_overlap_recall == 0:
        avg_overlap_f1 = 0.0
    else:
        avg_overlap_f1 = 2 * avg_overlap_precision * avg_overlap_recall / (avg_overlap_precision + avg_overlap_recall)

    # Compute classification metrics
    def compute_metrics(y_true, y_pred):
        return {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

    return {
        'start_time_normalization': compute_metrics(start_truths, start_preds),
        'end_time_normalization': compute_metrics(end_truths, end_preds),
        'event_time_pair_extraction': compute_metrics(pair_truths, pair_preds),
        'event_span_overlap': {
            'precision': avg_overlap_precision,
            'recall': avg_overlap_recall,
            'f1': avg_overlap_f1,
        }
    }