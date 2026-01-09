#!/usr/bin/env python3
"""
Extract event timelines from i2b2 training dataset.
Records sequences of events, event durations, and time between events.
Outputs to i2b2_patient_timelines.json in the same format as patient_timelines.json.
"""

import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import argparse
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
except ImportError as e:
    print(f"Error importing data loader: {e}")
    print("Make sure you're running this script from the correct environment.")
    sys.exit(1)


def extract_event_from_text(text, start_char, end_char):
    """Extract event text from document"""
    try:
        return text[start_char:end_char]
    except:
        return "unknown_event"


def get_event_type(row):
    """Determine event type from row data"""
    # You can enhance this based on your data structure
    # For now, we'll use a simple heuristic or return 'event'
    event_text = extract_event_from_text(row['text'], row['start_char'], row['end_char']).lower()

    # Simple type classification based on keywords
    if any(word in event_text for word in ['surgery', 'procedure', 'operation', 'biopsy', 'resection']):
        return 'procedure'
    elif any(word in event_text for word in ['test', 'scan', 'ct', 'mri', 'xray', 'lab', 'blood', 'ultrasound', 'mammogram']):
        return 'test'
    elif any(word in event_text for word in ['pain', 'fever', 'nausea', 'bleeding', 'symptom', 'ache', 'swelling']):
        return 'symptom'
    elif any(word in event_text for word in ['admission', 'discharge', 'visit', 'consultation', 'followup']):
        return 'occurrence'
    elif any(word in event_text for word in ['medication', 'drug', 'treatment', 'therapy', 'dose']):
        return 'treatment'
    else:
        return 'other'


def compute_time_between_events(prev_event, curr_event):
    """Compute time between two events in minutes"""
    try:
        prev_end = prev_event['end_time_minutes']
        curr_start = curr_event['start_time_minutes']
        return curr_start - prev_end
    except:
        return None


def compute_event_duration(event):
    """Compute event duration in minutes"""
    try:
        duration = event['end_time_minutes'] - event['start_time_minutes']
        return duration if duration > 0 else None
    except:
        return None


def format_duration_string(duration_minutes):
    """Format duration in a human-readable string"""
    if duration_minutes is None:
        return None

    if duration_minutes < 60:
        return f"{duration_minutes} minutes"
    elif duration_minutes < 1440:  # Less than a day
        hours = duration_minutes / 60
        return f"{hours:.1f} hours"
    else:
        days = duration_minutes / 1440
        return f"{days:.1f} days"


def extract_temporal_expression(row):
    """Extract temporal expression from row if available"""
    # This would require temporal expression extraction logic
    # For now, return None as we're working with absolute times
    return None


def extract_timelines_from_i2b2(output_path='rag_llm_approach/i2b2_patient_timelines.json'):
    """
    Extract event timelines from i2b2 training data.
    Groups events by document and creates timeline sequences.
    """
    print("Loading i2b2 training data...")
    df = load_i2b2_absolute_data(test_split=False)

    print(f"Loaded {len(df)} events from training data")

    # Group events by document
    document_groups = df.groupby('document_id')

    timelines = []
    total_documents = len(document_groups)

    print(f"Processing {total_documents} documents...")

    for doc_idx, (document_id, doc_events) in enumerate(document_groups, 1):
        # Sort events by start time
        doc_events = doc_events.sort_values('start_time_minutes')

        events = []
        timeline_sequence = []
        transitions = []

        prev_event_data = None

        for idx, (_, row) in enumerate(doc_events.iterrows()):
            event_id = f"evt_{idx + 1}"
            event_text = extract_event_from_text(row['text'], row['start_char'], row['end_char'])
            event_type = get_event_type(row)
            duration_minutes = compute_event_duration(row)
            duration_str = format_duration_string(duration_minutes)
            temporal_expr = extract_temporal_expression(row)

            # Create event data structure
            event_data = {
                'event_id': event_id,
                'event_text': event_text,
                'event_type': event_type,
                'temporal_expression': temporal_expr,
                'absolute_time': None,  # Will be formatted from minutes
                'start_time_minutes': int(row['start_time_minutes']),
                'end_time_minutes': int(row['end_time_minutes']),
                'sequence_position': idx,
                'concurrent_with': [],
                'duration': duration_str,
                'duration_minutes': duration_minutes,
                'confidence': 1.0
            }

            # Compute time between events
            if prev_event_data is not None:
                time_between = compute_time_between_events(prev_event_data, event_data)
                if time_between is not None and time_between >= 0:
                    transition = {
                        'from_event': prev_event_data['event_id'],
                        'to_event': event_id,
                        'time_between_minutes': time_between,
                        'time_between_formatted': format_duration_string(time_between)
                    }
                    transitions.append(transition)

            events.append(event_data)
            timeline_sequence.append(event_id)
            prev_event_data = event_data

        # Create timeline object
        timeline = {
            'document_id': document_id,
            'events': events,
            'transitions': transitions,
            'summary': {
                'total_events': len(events),
                'total_transitions': len(transitions),
                'timeline_duration_minutes': int(events[-1]['end_time_minutes'] - events[0]['start_time_minutes']) if events else 0,
                'timeline_duration_formatted': format_duration_string(
                    events[-1]['end_time_minutes'] - events[0]['start_time_minutes']
                ) if events else None
            },
            'timeline_sequence': timeline_sequence
        }

        timelines.append(timeline)

        if doc_idx % 10 == 0:
            print(f"Processed {doc_idx}/{total_documents} documents...")

    # Create output structure
    output_data = {
        'timelines': timelines,
        'metadata': {
            'total_timelines': len(timelines),
            'last_updated': datetime.now().isoformat(),
            'documents_processed': total_documents,
            'source': 'i2b2_training_data',
            'total_events': sum(t['summary']['total_events'] for t in timelines),
            'total_transitions': sum(t['summary']['total_transitions'] for t in timelines)
        }
    }

    # Save to file
    print(f"\nSaving timelines to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary statistics
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents processed: {total_documents}")
    print(f"Total timelines created: {len(timelines)}")
    print(f"Total events extracted: {output_data['metadata']['total_events']}")
    print(f"Total transitions recorded: {output_data['metadata']['total_transitions']}")

    # Calculate average statistics
    avg_events_per_timeline = output_data['metadata']['total_events'] / len(timelines) if timelines else 0
    avg_transitions_per_timeline = output_data['metadata']['total_transitions'] / len(timelines) if timelines else 0

    print(f"\nAverage events per timeline: {avg_events_per_timeline:.2f}")
    print(f"Average transitions per timeline: {avg_transitions_per_timeline:.2f}")

    # Event type distribution
    event_types = defaultdict(int)
    for timeline in timelines:
        for event in timeline['events']:
            event_types[event['event_type']] += 1

    print(f"\nEvent type distribution:")
    for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / output_data['metadata']['total_events']) * 100
        print(f"  {event_type}: {count} ({percentage:.1f}%)")

    # Duration statistics
    durations = [e['duration_minutes'] for t in timelines for e in t['events'] if e['duration_minutes'] is not None]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\nAverage event duration: {format_duration_string(avg_duration)}")
        print(f"Min duration: {format_duration_string(min(durations))}")
        print(f"Max duration: {format_duration_string(max(durations))}")

    print(f"\n{'='*60}")
    print(f"Timelines saved to: {output_path}")
    print(f"{'='*60}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description='Extract event timelines from i2b2 training data')
    parser.add_argument('--output', '-o',
                       default='rag_llm_approach/i2b2_patient_timelines.json',
                       help='Output file path (default: rag_llm_approach/i2b2_patient_timelines.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    try:
        output_data = extract_timelines_from_i2b2(output_path=args.output)

        if args.verbose:
            print("\nSample timeline (first document):")
            print(json.dumps(output_data['timelines'][0], indent=2)[:1000] + "...")

        return 0

    except Exception as e:
        print(f"Error extracting timelines: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

