import os
import pandas as pd
import cryptpandas as crp
from rag_llm_approach.build_knowledge_graph import DocumentSummary
from collections import defaultdict
from typing import List, Tuple, Dict

def extract_events_from_dataframe(df) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Extract all unique events from the dataframe grouped by document_id.
    Only includes events that are not of type TIMEX3.
    Returns a dictionary mapping document_id to list of (start, end, event_text) tuples.
    """
    events_by_doc = defaultdict(set)

    for _, row in df.iterrows():
        doc_id = row["document_id"]
        text = row["text"]

        # Extract event1 if it exists and is not TIMEX3
        if (pd.notna(row.get("event1_start")) and pd.notna(row.get("event1_end")) and
            pd.notna(row.get("event1_type")) and row.get("event1_type") != "TIMEX3"):
            start1 = int(row["event1_start"])
            end1 = int(row["event1_end"])
            event1_text = text[start1:end1]
            events_by_doc[doc_id].add((start1, end1, event1_text))

        # Extract event2 if it exists and is not TIMEX3
        if (pd.notna(row.get("event2_start")) and pd.notna(row.get("event2_end")) and
            pd.notna(row.get("event2_type")) and row.get("event2_type") != "TIMEX3"):
            start2 = int(row["event2_start"])
            end2 = int(row["event2_end"])
            event2_text = text[start2:end2]
            events_by_doc[doc_id].add((start2, end2, event2_text))

    # Convert sets to sorted lists (by start position) and return as proper dict
    result = {}
    for doc_id in events_by_doc:
        result[doc_id] = sorted(list(events_by_doc[doc_id]))

    return result

def create_xml_tagged_text(text: str, events: List[Tuple[int, int, str]]) -> str:
    """
    Create XML-tagged version of text with events marked.
    Events should be sorted by start position.
    """
    if not events:
        return text

    # Sort events by start position to process from end to beginning
    # (to avoid position shifts when inserting tags)
    sorted_events = sorted(events, key=lambda x: x[0], reverse=True)

    tagged_text = text
    event_id = len(sorted_events)

    for start, end, event_text in sorted_events:
        # Insert closing tag
        tagged_text = tagged_text[:end] + f"</event{event_id}>" + tagged_text[end:]
        # Insert opening tag
        tagged_text = tagged_text[:start] + f"<event{event_id}>" + tagged_text[start:]
        event_id -= 1

    return tagged_text

# Load the encrypted dataframe
df = crp.read_encrypted(path='thyme.crypt', password=os.environ['TP'])

# Extract events grouped by document
events_by_document = extract_events_from_dataframe(df)

# Create documents with both original and XML-tagged text
documents = []
document_content = []

for _, row in df.drop_duplicates(subset="document_id")[["document_id", "text"]].iterrows():
    doc_id = row["document_id"]
    original_text = row["text"]

    # Get events for this document
    doc_events = events_by_document.get(doc_id, [])

    # Create XML-tagged version
    xml_tagged_text = create_xml_tagged_text(original_text, doc_events)

    # Create DocumentSummary with additional information
    doc_summary = DocumentSummary(
        doc_id=doc_id,
        text=xml_tagged_text,  # Use XML-tagged version as main text
        annotations={
            'original_text': original_text,
            'events': doc_events,
            'num_events': len(doc_events)
        }
    )

    documents.append(doc_summary)
    document_content.append(xml_tagged_text)

print(f"Loaded {len(documents)} documents with XML-tagged events")
print(f"Total unique events across all documents: {sum(len(events) for events in events_by_document.values())}")

# Print sample of first document with events for verification
if documents:
    sample_doc = documents[0]
    print(f"\nSample document {sample_doc.doc_id}:")
    print(f"Number of events: {sample_doc.annotations['num_events']}")
    if sample_doc.annotations['events']:
        print("Events found:")
        for i, (start, end, event_text) in enumerate(sample_doc.annotations['events'][:3], 1):
            print(f"  Event {i}: '{event_text}' (pos: {start}-{end})")
    print(f"Tagged text preview: {sample_doc.text[:200]}...")
