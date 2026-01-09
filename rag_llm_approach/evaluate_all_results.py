#!/usr/bin/env python3
"""
Evaluate all RAG model results from JSON files.
Computes accuracy metrics for start time, end time, and duration predictions.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import calendar
from typing import Dict, List, Tuple
import glob
import numpy as np

# Reference date: 1900-01-01 00:00:00
REFERENCE = datetime(1900, 1, 1, 0, 0, 0)


def expand_bounds(lower_minutes: int, upper_minutes: int) -> Tuple[int, int]:
    """
    Expands lower and upper bounds (in minutes since 1900-01-01 00:00)
    to the nearest day, week, month, year, or decade depending on their range.
    """
    # Convert minutes since reference into datetime objects
    lower_dt = REFERENCE + timedelta(minutes=lower_minutes)
    upper_dt = REFERENCE + timedelta(minutes=upper_minutes)

    # If already ordered incorrectly, swap
    if lower_dt > upper_dt:
        lower_dt, upper_dt = upper_dt, lower_dt

    # Helper functions for snapping
    def expand_to_day(dt):
        return dt.replace(hour=0, minute=0), dt.replace(hour=23, minute=59)

    def expand_to_week(dt):
        # ISO weeks: Monday=0, Sunday=6
        start = dt - timedelta(days=dt.weekday())
        start = start.replace(hour=0, minute=0)
        end = start + timedelta(days=6, hours=23, minutes=59)
        return start, end

    def expand_to_month(dt):
        start = dt.replace(day=1, hour=0, minute=0)
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        end = dt.replace(day=last_day, hour=23, minute=59)
        return start, end

    def expand_to_year(dt):
        start = dt.replace(month=1, day=1, hour=0, minute=0)
        end = dt.replace(month=12, day=31, hour=23, minute=59)
        return start, end

    def expand_to_decade(dt):
        start_year = (dt.year // 10) * 10
        start = datetime(start_year, 1, 1, 0, 0)
        end = datetime(start_year + 9, 12, 31, 23, 59)
        return start, end

    # Check containment levels
    if lower_dt.date() == upper_dt.date():
        lower_exp, upper_exp = expand_to_day(lower_dt)
    elif (lower_dt.isocalendar()[0:2] == upper_dt.isocalendar()[0:2]):  # same ISO week
        lower_exp, upper_exp = expand_to_week(lower_dt)
    elif lower_dt.year == upper_dt.year and lower_dt.month == upper_dt.month:
        lower_exp, upper_exp = expand_to_month(lower_dt)
    elif lower_dt.year == upper_dt.year:
        lower_exp, upper_exp = expand_to_year(lower_dt)
    elif lower_dt.year // 10 == upper_dt.year // 10:
        lower_exp, upper_exp = expand_to_decade(lower_dt)
    else:
        # If they are far apart (different decades), keep them as-is
        lower_exp, upper_exp = lower_dt, upper_dt

    # Convert back to minutes since reference
    lower_minutes_exp = int((lower_exp - REFERENCE).total_seconds() // 60)
    upper_minutes_exp = int((upper_exp - REFERENCE).total_seconds() // 60)

    return lower_minutes_exp, upper_minutes_exp


def parse_iso_to_minutes(iso_string: str) -> float:
    """Convert ISO format datetime string to minutes since 1900-01-01."""
    try:
        dt = datetime.fromisoformat(iso_string)
        minutes = (dt - REFERENCE).total_seconds() / 60
        return minutes
    except Exception as e:
        print(f"Error parsing datetime '{iso_string}': {e}")
        return None


def is_within_bounds(predicted_minutes: float, lower_minutes: float, upper_minutes: float) -> bool:
    """Check if predicted value falls within the bounds."""
    if predicted_minutes is None:
        return False
    return lower_minutes <= predicted_minutes <= upper_minutes


def bootstrap_confidence_interval(correct_count: int, total_count: int, n_bootstrap: int = 1000, confidence_level: float = 0.95) -> float:
    """
    Calculate confidence interval for accuracy using bootstrap resampling.

    Args:
        correct_count: Number of correct predictions
        total_count: Total number of predictions
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Half-width of the confidence interval (the ± value)
    """
    if total_count == 0:
        return 0.0

    # Create binary array: 1 for correct, 0 for incorrect
    data = np.array([1] * correct_count + [0] * (total_count - correct_count))

    # Bootstrap resampling
    bootstrap_accuracies = []
    rng = np.random.RandomState(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(data, size=len(data), replace=True)
        # Calculate accuracy for this sample
        accuracy = np.mean(sample) * 100
        bootstrap_accuracies.append(accuracy)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
    ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)

    # Return half-width (± value)
    original_accuracy = (correct_count / total_count) * 100
    ci_half_width = (ci_upper - ci_lower) / 2

    return ci_half_width


def evaluate_predictions(results: List[Dict]) -> Dict[str, float]:
    """
    Evaluate predictions and compute accuracy metrics.

    Returns a dictionary with:
    - start_accuracy: % of predictions within expanded start bounds
    - end_accuracy: % of predictions within expanded end bounds
    - duration_accuracy: % of predictions within expanded duration bounds
    - total_predictions: total number of predictions
    - valid_predictions: predictions with valid start and end times
    """
    total = len(results)

    start_correct = 0
    end_correct = 0
    duration_correct = 0
    valid_count = 0

    for result in results:
        try:
            llm_response = result.get('llm_response', {})
            ground_truth = result.get('ground_truth', {})

            # Parse predicted start and end times
            pred_start_str = llm_response.get('start')
            pred_end_str = llm_response.get('end')

            if not pred_start_str or not pred_end_str:
                continue

            pred_start_minutes = parse_iso_to_minutes(pred_start_str)
            pred_end_minutes = parse_iso_to_minutes(pred_end_str)

            valid_count += 1
            if pred_start_minutes is None or pred_end_minutes is None:
                continue


            # Calculate predicted duration
            pred_duration_minutes = pred_end_minutes - pred_start_minutes

            # Get ground truth bounds
            start_lower = ground_truth.get('start_lower_minutes')
            start_upper = ground_truth.get('start_upper_minutes')
            end_lower = ground_truth.get('end_lower_minutes')
            end_upper = ground_truth.get('end_upper_minutes')
            duration_lower = ground_truth.get('duration_lower_minutes')
            duration_upper = ground_truth.get('duration_upper_minutes')

            # Expand bounds
            if start_lower is not None and start_upper is not None:
                start_lower_exp, start_upper_exp = expand_bounds(start_lower, start_upper)
                if is_within_bounds(pred_start_minutes, start_lower_exp, start_upper_exp):
                    start_correct += 1

            if end_lower is not None and end_upper is not None:
                end_lower_exp, end_upper_exp = expand_bounds(end_lower, end_upper)
                if is_within_bounds(pred_end_minutes, end_lower_exp, end_upper_exp):
                    end_correct += 1

            if duration_lower is not None and duration_upper is not None:
                duration_lower_exp, duration_upper_exp = expand_bounds(duration_lower, duration_upper)
                if is_within_bounds(pred_duration_minutes, duration_lower_exp, duration_upper_exp):
                    duration_correct += 1

        except Exception as e:
            print(f"Error processing result for document {result.get('document_id', 'unknown')}: {e}")
            continue

    # Compute percentages
    start_accuracy = (start_correct / valid_count * 100) if valid_count > 0 else 0.0
    end_accuracy = (end_correct / valid_count * 100) if valid_count > 0 else 0.0
    duration_accuracy = (duration_correct / valid_count * 100) if valid_count > 0 else 0.0
    overall_accuracy = (start_accuracy + end_accuracy) / 2.0

    # Compute 95% confidence intervals using bootstrap
    start_ci = bootstrap_confidence_interval(start_correct, valid_count) if valid_count > 0 else 0.0
    end_ci = bootstrap_confidence_interval(end_correct, valid_count) if valid_count > 0 else 0.0
    duration_ci = bootstrap_confidence_interval(duration_correct, valid_count) if valid_count > 0 else 0.0

    # For overall accuracy, we need to bootstrap the combined metric
    # We'll use the average of start and end CIs as an approximation
    overall_ci = (start_ci + end_ci) / 2.0

    return {
        'start_accuracy': start_accuracy,
        'end_accuracy': end_accuracy,
        'duration_accuracy': duration_accuracy,
        'overall_accuracy': overall_accuracy,
        'start_ci': start_ci,
        'end_ci': end_ci,
        'duration_ci': duration_ci,
        'overall_ci': overall_ci,
        'total_predictions': total,
        'valid_predictions': valid_count,
        'start_correct': start_correct,
        'end_correct': end_correct,
        'duration_correct': duration_correct
    }


def read_all_results(results_dir: str = 'rag_llm_approach/results') -> Dict[str, List[Dict]]:
    """Read all JSON result files from the results directory."""
    results_dict = {}

    # Check if directory exists
    if not os.path.exists(results_dir):
        # Try alternative directory (current rag_llm_approach directory)
        alt_dir = 'rag_llm_approach'
        if os.path.exists(alt_dir):
            print(f"Warning: {results_dir} not found, using {alt_dir} instead")
            results_dir = alt_dir
        else:
            print(f"Warning: Directory {results_dir} does not exist")
            return results_dict

    # Find all JSON files that look like prediction results
    # Look for files matching pattern: *predictions*.json or rag_*.json
    json_files = []
    json_files.extend(glob.glob(os.path.join(results_dir, '*predictions*.json')))
    json_files.extend(glob.glob(os.path.join(results_dir, 'rag_*.json')))

    # Remove duplicates
    json_files = list(set(json_files))

    if not json_files:
        print(f"Warning: No prediction JSON files found in {results_dir}")
        print(f"Looking for files matching: *predictions*.json or rag_*.json")
        return results_dict

    for json_file in json_files:
        model_name = Path(json_file).stem
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)

            # Verify it's a valid results file (should be a list of dicts with expected structure)
            if isinstance(results, list) and len(results) > 0:
                # Check if first item has expected structure
                first_item = results[0]
                if 'llm_response' in first_item and 'ground_truth' in first_item:
                    results_dict[model_name] = results
                    print(f"✓ Loaded {len(results)} predictions from {model_name}")
                else:
                    print(f"⚠ Skipping {json_file}: doesn't have expected structure")
            else:
                print(f"⚠ Skipping {json_file}: not a valid results list")
        except Exception as e:
            print(f"✗ Error reading {json_file}: {e}")

    return results_dict


def generate_markdown_table(evaluation_results: Dict[str, Dict]) -> str:
    """Generate a markdown table from evaluation results."""
    lines = []
    lines.append("# RAG Model Evaluation Results")
    lines.append("")
    lines.append("| Model | Overall Accuracy (%) | Start Accuracy (%) | End Accuracy (%) | Duration Accuracy (%) | Valid/Total Predictions |")
    lines.append("|-------|---------------------|--------------------|------------------|----------------------|------------------------|")

    # Sort by overall accuracy (descending)
    sorted_results = sorted(evaluation_results.items(),
                           key=lambda x: x[1]['overall_accuracy'],
                           reverse=True)

    for model_name, metrics in sorted_results:
        valid = metrics['valid_predictions']
        total = metrics['total_predictions']
        lines.append(
            f"| {model_name} | "
            f"{metrics['overall_accuracy']:.2f} ± {metrics['overall_ci']:.2f} | "
            f"{metrics['start_accuracy']:.2f} ± {metrics['start_ci']:.2f} | "
            f"{metrics['end_accuracy']:.2f} ± {metrics['end_ci']:.2f} | "
            f"{metrics['duration_accuracy']:.2f} ± {metrics['duration_ci']:.2f} | "
            f"{valid}/{total} |"
        )

    return "\n".join(lines)


def generate_latex_table(evaluation_results: Dict[str, Dict]) -> str:
    """Generate a LaTeX table from evaluation results."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{RAG Model Evaluation Results}")
    lines.append("\\label{tab:rag_results}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Model} & \\textbf{Overall Acc. (\\%)} & \\textbf{Start Acc. (\\%)} & \\textbf{End Acc. (\\%)} & \\textbf{Duration Acc. (\\%)} & \\textbf{Valid/Total} \\\\")
    lines.append("\\hline")

    # Sort by overall accuracy (descending)
    sorted_results = sorted(evaluation_results.items(),
                           key=lambda x: x[1]['overall_accuracy'],
                           reverse=True)

    for model_name, metrics in sorted_results:
        # Escape underscores for LaTeX
        model_name_latex = model_name.replace('_', '\\_')
        valid = metrics['valid_predictions']
        total = metrics['total_predictions']
        lines.append(
            f"{model_name_latex} & "
            f"{metrics['overall_accuracy']:.2f} $\\pm$ {metrics['overall_ci']:.2f} & "
            f"{metrics['start_accuracy']:.2f} $\\pm$ {metrics['start_ci']:.2f} & "
            f"{metrics['end_accuracy']:.2f} $\\pm$ {metrics['end_ci']:.2f} & "
            f"{metrics['duration_accuracy']:.2f} $\\pm$ {metrics['duration_ci']:.2f} & "
            f"{valid}/{total} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate RAG model predictions')
    parser.add_argument('--results_dir', type=str, default='rag_llm_approach/results',
                       help='Directory containing result JSON files (default: rag_llm_approach/results)')
    args = parser.parse_args()

    print("="*80)
    print("RAG Model Evaluation Script")
    print("="*80)
    print()

    # Read all results
    print(f"Reading results from: {args.results_dir}")
    all_results = read_all_results(args.results_dir)

    if not all_results:
        print("\n❌ No results found. Please ensure:")
        print("   1. The directory exists")
        print("   2. JSON files are present")
        print("   3. Files match pattern: *predictions*.json or rag_*.json")
        print("\nYou can specify a custom directory with: --results_dir <path>")
        return

    print()
    print("="*80)
    print("Evaluating Models")
    print("="*80)
    print()

    # Evaluate each model
    evaluation_results = {}
    for model_name, results in all_results.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_predictions(results)
        evaluation_results[model_name] = metrics

        print(f"  Total predictions: {metrics['total_predictions']}")
        print(f"  Valid predictions: {metrics['valid_predictions']}")
        print(f"  Overall accuracy: {metrics['overall_accuracy']:.2f} ± {metrics['overall_ci']:.2f}%")
        print(f"  Start accuracy: {metrics['start_accuracy']:.2f} ± {metrics['start_ci']:.2f}% ({metrics['start_correct']}/{metrics['valid_predictions']})")
        print(f"  End accuracy: {metrics['end_accuracy']:.2f} ± {metrics['end_ci']:.2f}% ({metrics['end_correct']}/{metrics['valid_predictions']})")
        print(f"  Duration accuracy: {metrics['duration_accuracy']:.2f} ± {metrics['duration_ci']:.2f}% ({metrics['duration_correct']}/{metrics['valid_predictions']})")

    print()
    print("="*80)
    print("Results Summary")
    print("="*80)
    print()

    # Generate markdown table
    markdown_table = generate_markdown_table(evaluation_results)
    print(markdown_table)

    # Save markdown table
    markdown_file = 'rag_llm_approach/evaluation_results.md'
    with open(markdown_file, 'w') as f:
        f.write(markdown_table)
    print(f"\n✓ Markdown table saved to: {markdown_file}")

    # Generate LaTeX table
    latex_table = generate_latex_table(evaluation_results)
    print()
    print("="*80)
    print("LaTeX Table")
    print("="*80)
    print()
    print(latex_table)

    # Save LaTeX table
    latex_file = 'rag_llm_approach/evaluation_results.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"\n✓ LaTeX table saved to: {latex_file}")

    # Save detailed results as JSON
    detailed_results_file = 'rag_llm_approach/evaluation_detailed_results.json'
    with open(detailed_results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"✓ Detailed results saved to: {detailed_results_file}")

    print()
    print("="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

