from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(predicted_starts, predicted_ends, predicted_durations, gold_starts, gold_ends, gold_durations):
    assert len(predicted_starts) == len(predicted_ends) == len(gold_starts) == len(gold_ends), "All lists must be of equal length."
    correct_start = 0
    correct_end = 0
    correct_duration = 0
    n = 0
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

        if gold_starts[i][1] <= pred_start_time <= gold_starts[i][2]:
            correct_start += 1
        if gold_ends[i][1] <= pred_end_time <= gold_ends[i][2]:
            correct_end += 1

        if gold_durations[i][1] <= pred_dur_time <= gold_durations[i][2]:
            correct_duration += 1
        n += 1

    return {
        "correct_start": correct_start,
        "correct_end": correct_end,
        "correct_duration": correct_duration,
        "n": n,
        "precision_start": correct_start / n,
        "precision_end": correct_end / n,
        "precision_duration": correct_duration / n
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