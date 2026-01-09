#!/usr/bin/env python3
"""
Analyze retrieval statistics from RAG pipeline results.
Provides comprehensive statistics on similar event retrieval performance.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter, defaultdict


def load_retrieval_stats(file_path: str) -> Dict:
    """Load retrieval statistics from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def analyze_single_file(stats: Dict) -> Dict:
    """
    Analyze a single retrieval statistics file.

    Returns detailed statistics including:
    - Basic counts and percentages
    - Distribution of similar events found
    - Context event statistics
    - Per-document statistics
    """
    if not stats:
        return None

    config_name = stats.get('configuration', 'Unknown')
    total_events = stats.get('total_events_processed', 0)
    events_with_similar = stats.get('events_with_similar_found', 0)
    events_without_similar = stats.get('events_without_similar', 0)
    percentage_with_similar = stats.get('percentage_with_similar', 0.0)
    avg_similar_events = stats.get('average_similar_events_per_query', 0.0)

    detailed_stats = stats.get('detailed_stats', [])

    # Distribution of similar events found
    similar_counts = []
    context_event_counts = []
    doc_event_counts = defaultdict(int)
    doc_similar_found = defaultdict(int)

    for event_stat in detailed_stats:
        num_similar = event_stat.get('num_similar_events_found', 0)
        similar_counts.append(num_similar)
        doc_id = event_stat.get('document_id')
        doc_event_counts[doc_id] += 1

        if num_similar > 0:
            doc_similar_found[doc_id] += 1

        # Extract context event counts
        retrieval_info = event_stat.get('retrieval_info', {})
        similar_events = retrieval_info.get('similar_events', [])
        for similar_event in similar_events:
            num_context = similar_event.get('num_context_events', 0)
            context_event_counts.append(num_context)

    # Calculate distribution statistics
    similar_counts_array = np.array(similar_counts)

    distribution = Counter(similar_counts)

    # Calculate percentiles for non-zero similar events
    non_zero_similar = [x for x in similar_counts if x > 0]

    # Calculate per-document statistics
    total_docs = len(doc_event_counts)
    docs_with_all_similar = sum(1 for doc_id in doc_event_counts
                                 if doc_similar_found[doc_id] == doc_event_counts[doc_id])
    docs_with_some_similar = sum(1 for doc_id in doc_similar_found if doc_similar_found[doc_id] > 0)
    docs_with_no_similar = total_docs - docs_with_some_similar

    # Context event statistics
    context_stats = {}
    if context_event_counts:
        context_array = np.array(context_event_counts)
        context_stats = {
            'mean': float(np.mean(context_array)),
            'median': float(np.median(context_array)),
            'std': float(np.std(context_array)),
            'min': int(np.min(context_array)),
            'max': int(np.max(context_array)),
            'total_context_events': int(np.sum(context_array))
        }

    return {
        'configuration': config_name,
        'total_events': total_events,
        'events_with_similar': events_with_similar,
        'events_without_similar': events_without_similar,
        'percentage_with_similar': percentage_with_similar,
        'average_similar_per_event': avg_similar_events,
        'similar_events_distribution': dict(distribution),
        'similar_events_stats': {
            'mean': float(np.mean(similar_counts_array)),
            'median': float(np.median(similar_counts_array)),
            'std': float(np.std(similar_counts_array)),
            'min': int(np.min(similar_counts_array)),
            'max': int(np.max(similar_counts_array)),
            'percentile_25': float(np.percentile(similar_counts_array, 25)),
            'percentile_75': float(np.percentile(similar_counts_array, 75)),
            'percentile_90': float(np.percentile(similar_counts_array, 90)),
            'percentile_95': float(np.percentile(similar_counts_array, 95)),
        },
        'non_zero_similar_stats': {
            'count': len(non_zero_similar),
            'mean': float(np.mean(non_zero_similar)) if non_zero_similar else 0.0,
            'median': float(np.median(non_zero_similar)) if non_zero_similar else 0.0,
            'std': float(np.std(non_zero_similar)) if non_zero_similar else 0.0,
            'min': int(np.min(non_zero_similar)) if non_zero_similar else 0,
            'max': int(np.max(non_zero_similar)) if non_zero_similar else 0,
        },
        'context_events_stats': context_stats,
        'document_stats': {
            'total_documents': total_docs,
            'docs_with_all_events_having_similar': docs_with_all_similar,
            'docs_with_some_events_having_similar': docs_with_some_similar,
            'docs_with_no_events_having_similar': docs_with_no_similar,
            'percentage_docs_with_all_similar': (docs_with_all_similar / total_docs * 100) if total_docs > 0 else 0.0,
            'percentage_docs_with_some_similar': (docs_with_some_similar / total_docs * 100) if total_docs > 0 else 0.0,
        }
    }


def find_retrieval_stats_files(results_dir: str = 'rag_llm_approach/results') -> List[str]:
    """Find all retrieval_stats_*.json files in the results directory."""
    if not os.path.exists(results_dir):
        alt_dir = 'rag_llm_approach'
        if os.path.exists(alt_dir):
            print(f"Warning: {results_dir} not found, using {alt_dir} instead")
            results_dir = alt_dir
        else:
            print(f"Warning: Directory {results_dir} does not exist")
            return []

    # Find all retrieval_stats_*.json files
    pattern = os.path.join(results_dir, 'retrieval_stats_*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"Warning: No retrieval_stats_*.json files found in {results_dir}")

    return files


def generate_markdown_summary(all_analyses: Dict[str, Dict]) -> str:
    """Generate a comprehensive markdown summary of all analyses."""
    lines = []
    lines.append("# Retrieval Statistics Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")

    # Summary table
    lines.append("| Configuration | Total Events | Events with Similar | % with Similar | Avg Similar/Event | Total Docs | Docs with All Similar |")
    lines.append("|--------------|--------------|---------------------|----------------|-------------------|------------|----------------------|")

    # Sort by percentage with similar (descending)
    sorted_analyses = sorted(all_analyses.items(),
                            key=lambda x: x[1]['percentage_with_similar'],
                            reverse=True)

    for file_name, analysis in sorted_analyses:
        config = analysis['configuration']
        total = analysis['total_events']
        with_similar = analysis['events_with_similar']
        pct_similar = analysis['percentage_with_similar']
        avg_similar = analysis['average_similar_per_event']
        total_docs = analysis['document_stats']['total_documents']
        docs_all_similar = analysis['document_stats']['docs_with_all_events_having_similar']

        lines.append(
            f"| {config} | {total} | {with_similar} | {pct_similar:.2f}% | {avg_similar:.2f} | {total_docs} | {docs_all_similar} |"
        )

    lines.append("")
    lines.append("## Detailed Statistics by Configuration")
    lines.append("")

    for file_name, analysis in sorted_analyses:
        lines.append(f"### {analysis['configuration']}")
        lines.append("")

        # Basic stats
        lines.append("#### Event-Level Statistics")
        lines.append("")
        lines.append(f"- **Total events processed**: {analysis['total_events']:,}")
        lines.append(f"- **Events with similar found**: {analysis['events_with_similar']:,} ({analysis['percentage_with_similar']:.2f}%)")
        lines.append(f"- **Events without similar**: {analysis['events_without_similar']:,}")
        lines.append(f"- **Average similar events per query**: {analysis['average_similar_per_event']:.2f}")
        lines.append("")

        # Similar events distribution
        lines.append("#### Similar Events Distribution")
        lines.append("")
        sim_stats = analysis['similar_events_stats']
        lines.append(f"- **Mean**: {sim_stats['mean']:.2f}")
        lines.append(f"- **Median**: {sim_stats['median']:.2f}")
        lines.append(f"- **Std Dev**: {sim_stats['std']:.2f}")
        lines.append(f"- **Range**: [{sim_stats['min']}, {sim_stats['max']}]")
        lines.append(f"- **25th percentile**: {sim_stats['percentile_25']:.2f}")
        lines.append(f"- **75th percentile**: {sim_stats['percentile_75']:.2f}")
        lines.append(f"- **90th percentile**: {sim_stats['percentile_90']:.2f}")
        lines.append(f"- **95th percentile**: {sim_stats['percentile_95']:.2f}")
        lines.append("")

        # Distribution breakdown
        lines.append("**Events by number of similar events found:**")
        lines.append("")
        dist = analysis['similar_events_distribution']
        for num_similar in sorted(dist.keys(), key=int):
            count = dist[num_similar]
            pct = (count / analysis['total_events'] * 100) if analysis['total_events'] > 0 else 0
            lines.append(f"- {num_similar} similar events: {count:,} events ({pct:.2f}%)")
        lines.append("")

        # Non-zero stats
        if analysis['non_zero_similar_stats']['count'] > 0:
            lines.append("#### Statistics for Events with Similar Found (non-zero)")
            lines.append("")
            nz_stats = analysis['non_zero_similar_stats']
            lines.append(f"- **Count**: {nz_stats['count']:,}")
            lines.append(f"- **Mean**: {nz_stats['mean']:.2f}")
            lines.append(f"- **Median**: {nz_stats['median']:.2f}")
            lines.append(f"- **Std Dev**: {nz_stats['std']:.2f}")
            lines.append(f"- **Range**: [{nz_stats['min']}, {nz_stats['max']}]")
            lines.append("")

        # Context events stats
        if analysis['context_events_stats']:
            lines.append("#### Context Events Statistics")
            lines.append("")
            ctx_stats = analysis['context_events_stats']
            lines.append(f"- **Total context events retrieved**: {ctx_stats['total_context_events']:,}")
            lines.append(f"- **Mean per similar event**: {ctx_stats['mean']:.2f}")
            lines.append(f"- **Median**: {ctx_stats['median']:.2f}")
            lines.append(f"- **Std Dev**: {ctx_stats['std']:.2f}")
            lines.append(f"- **Range**: [{ctx_stats['min']}, {ctx_stats['max']}]")
            lines.append("")

        # Document-level stats
        lines.append("#### Document-Level Statistics")
        lines.append("")
        doc_stats = analysis['document_stats']
        lines.append(f"- **Total documents**: {doc_stats['total_documents']}")
        lines.append(f"- **Documents with all events having similar**: {doc_stats['docs_with_all_events_having_similar']} ({doc_stats['percentage_docs_with_all_similar']:.2f}%)")
        lines.append(f"- **Documents with some events having similar**: {doc_stats['docs_with_some_events_having_similar']} ({doc_stats['percentage_docs_with_some_similar']:.2f}%)")
        lines.append(f"- **Documents with no events having similar**: {doc_stats['docs_with_no_events_having_similar']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_latex_table(all_analyses: Dict[str, Dict]) -> str:
    """Generate a LaTeX table summarizing retrieval statistics."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Retrieval Statistics Summary}")
    lines.append("\\label{tab:retrieval_stats}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Configuration} & \\textbf{Total} & \\textbf{Events with} & \\textbf{\\% with} & \\textbf{Avg Similar} & \\textbf{Median} & \\textbf{Max} \\\\")
    lines.append("                      & \\textbf{Events} & \\textbf{Similar} & \\textbf{Similar} & \\textbf{per Event} & \\textbf{Similar} & \\textbf{Similar} \\\\")
    lines.append("\\hline")

    # Sort by percentage with similar (descending)
    sorted_analyses = sorted(all_analyses.items(),
                            key=lambda x: x[1]['percentage_with_similar'],
                            reverse=True)

    for file_name, analysis in sorted_analyses:
        # Extract configuration name and escape underscores
        config = analysis['configuration'].replace('_', '\\_')
        # Shorten if too long
        if len(config) > 40:
            config = config[:37] + "..."

        total = analysis['total_events']
        with_similar = analysis['events_with_similar']
        pct_similar = analysis['percentage_with_similar']
        avg_similar = analysis['average_similar_per_event']
        median = analysis['similar_events_stats']['median']
        max_val = analysis['similar_events_stats']['max']

        # For no-RAG configurations (0% with similar), show N/A for retrieval stats
        if pct_similar == 0.0:
            lines.append(
                f"{config} & {total:,} & -- & -- & -- & -- & -- \\\\"
            )
        else:
            lines.append(
                f"{config} & {total:,} & {with_similar:,} & {pct_similar:.1f}\\% & {avg_similar:.2f} & {median:.0f} & {max_val} \\\\"
            )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_comparison_table(all_analyses: Dict[str, Dict]) -> str:
    """Generate a detailed comparison markdown table."""
    lines = []
    lines.append("## Comparison Table: Similar Events Retrieved")
    lines.append("")
    lines.append("| Configuration | Mean | Median | Std | Min | Max | 25th % | 75th % | 95th % |")
    lines.append("|--------------|------|--------|-----|-----|-----|--------|--------|--------|")

    sorted_analyses = sorted(all_analyses.items(),
                            key=lambda x: x[1]['similar_events_stats']['mean'],
                            reverse=True)

    for file_name, analysis in sorted_analyses:
        config = analysis['configuration']
        stats = analysis['similar_events_stats']

        lines.append(
            f"| {config} | {stats['mean']:.2f} | {stats['median']:.2f} | "
            f"{stats['std']:.2f} | {stats['min']} | {stats['max']} | "
            f"{stats['percentile_25']:.2f} | {stats['percentile_75']:.2f} | "
            f"{stats['percentile_95']:.2f} |"
        )

    return "\n".join(lines)


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze RAG retrieval statistics')
    parser.add_argument('--results_dir', type=str, default='rag_llm_approach/results',
                       help='Directory containing retrieval_stats_*.json files')
    args = parser.parse_args()

    print("="*80)
    print("Retrieval Statistics Analysis Script")
    print("="*80)
    print()

    # Find all retrieval stats files
    print(f"Searching for retrieval stats files in: {args.results_dir}")
    stats_files = find_retrieval_stats_files(args.results_dir)

    if not stats_files:
        print("\n❌ No retrieval stats files found. Please ensure:")
        print("   1. The directory exists")
        print("   2. Files matching pattern 'retrieval_stats_*.json' are present")
        print("\nYou can specify a custom directory with: --results_dir <path>")
        return

    print(f"✓ Found {len(stats_files)} retrieval stats files")
    print()

    # Analyze each file
    print("="*80)
    print("Analyzing Retrieval Statistics")
    print("="*80)
    print()

    all_analyses = {}

    for stats_file in stats_files:
        file_name = Path(stats_file).stem
        print(f"Processing: {file_name}")

        stats = load_retrieval_stats(stats_file)
        if stats:
            analysis = analyze_single_file(stats)
            if analysis:
                all_analyses[file_name] = analysis
                print(f"  ✓ Analyzed {analysis['total_events']:,} events")
                print(f"    - {analysis['percentage_with_similar']:.2f}% with similar events found")
                print(f"    - Average {analysis['average_similar_per_event']:.2f} similar events per query")
        print()

    if not all_analyses:
        print("❌ No valid analyses generated")
        return

    print("="*80)
    print("Generating Reports")
    print("="*80)
    print()

    # Generate markdown summary
    markdown_summary = generate_markdown_summary(all_analyses)

    # Save markdown summary
    markdown_file = 'rag_llm_approach/retrieval_analysis.md'
    with open(markdown_file, 'w') as f:
        f.write(markdown_summary)
    print(f"✓ Markdown summary saved to: {markdown_file}")

    # Generate comparison table
    comparison_table = generate_comparison_table(all_analyses)

    # Append to markdown file
    with open(markdown_file, 'a') as f:
        f.write("\n\n")
        f.write(comparison_table)

    # Generate LaTeX table
    latex_table = generate_latex_table(all_analyses)
    latex_file = 'rag_llm_approach/retrieval_analysis.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_file}")

    # Save detailed results as JSON
    detailed_file = 'rag_llm_approach/retrieval_analysis_detailed.json'
    with open(detailed_file, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    print(f"✓ Detailed analysis saved to: {detailed_file}")

    print()
    print("="*80)
    print("Summary Preview (Markdown)")
    print("="*80)
    print()
    print(markdown_summary)

    print()
    print("="*80)
    print("LaTeX Table Preview")
    print("="*80)
    print()
    print(latex_table)

    print()
    print("="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

