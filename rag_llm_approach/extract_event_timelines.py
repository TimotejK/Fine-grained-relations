#!/usr/bin/env python3
"""
Event Timeline Extraction from Medical Discharge Summaries

This program extracts sequences of events (timelines) from medical discharge summaries,
capturing the temporal ordering of events, concurrent events, and time durations.
It supports resumable processing with automatic checkpoint saving.
"""

import json
import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

# Third-party imports
import openai
import requests
from pydantic import BaseModel, Field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pydantic models for structured timeline output
class TimelineEventType(str, Enum):
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"
    DIAGNOSIS = "diagnosis"
    TEST = "test"
    ADMISSION = "admission"
    DISCHARGE = "discharge"
    OTHER = "other"


class TimelineEvent(BaseModel):
    """Represents a single event in the timeline"""
    event_id: str = Field(description="Unique identifier for this event")
    event_text: str = Field(description="Exact event description from the text")
    event_type: TimelineEventType = Field(description="Type of medical event")
    temporal_expression: Optional[str] = Field(None, description="Exact temporal phrase from text")
    absolute_time: Optional[str] = Field(None, description="Absolute time if available (e.g., '2024-01-15 14:30')")
    sequence_position: int = Field(description="Position in the timeline sequence (0-based)")
    concurrent_with: List[str] = Field(default_factory=list, description="IDs of events happening at the same time")
    duration: Optional[str] = Field(None, description="How long the event lasted")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)


class TimelineTransition(BaseModel):
    """Represents the transition between two events in the timeline"""
    from_event_id: str = Field(description="ID of the earlier event")
    to_event_id: str = Field(description="ID of the later event")
    time_elapsed: Optional[str] = Field(None, description="Time that passed between events (e.g., '2 days', '3 hours')")
    transition_type: str = Field(description="Type of transition: 'sequential', 'concurrent', 'overlapping'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)


class PatientTimeline(BaseModel):
    """Represents the complete timeline for a patient"""
    document_id: str = Field(description="Document identifier")
    events: List[TimelineEvent] = Field(description="Ordered list of events in chronological sequence")
    transitions: List[TimelineTransition] = Field(description="Transitions between events")
    summary: Optional[str] = Field(None, description="Brief summary of the timeline")

    # Note: timeline_sequence is not part of the schema for OpenAI API compatibility
    # It will be constructed post-hoc using _construct_timeline_sequence()


@dataclass
class DocumentSummary:
    """Document structure for input"""
    doc_id: str
    text: str
    annotations: List[Dict] = None


class LLMInterface:
    """Interface for different LLM providers with structured output support"""

    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4"
        elif self.provider == "ollama":
            self.base_url = "http://localhost:11434/api/generate"
            self.model = model or "gemma3:27b"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_structured_timeline(self, prompt: str, max_tokens: int = 3000) -> PatientTimeline:
        """Generate structured timeline using LLM with proper schema validation"""
        response_text = ""

        try:
            if self.provider == "openai":
                # Use OpenAI's structured output
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a medical NLP expert specializing in temporal event extraction. "
                                "Your task is to extract complete timelines of patient events from medical texts. "
                                "Create chronologically ordered sequences showing what happened to the patient, "
                                "when it happened, and how long things took. "
                                "Identify concurrent events and time intervals between events when possible."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format=PatientTimeline,
                )
                return response.choices[0].message.parsed

            elif self.provider == "ollama":
                # For Ollama, use enhanced prompt with JSON schema
                enhanced_prompt = f"""
{prompt}

IMPORTANT: Respond with valid JSON matching this exact schema:
{{
    "document_id": "string",
    "events": [
        {{
            "event_id": "evt_1",
            "event_text": "string",
            "event_type": "medication|procedure|symptom|diagnosis|test|admission|discharge|other",
            "temporal_expression": "string or null",
            "absolute_time": "string or null",
            "sequence_position": integer,
            "concurrent_with": ["evt_2", "evt_3"],
            "duration": "string or null",
            "confidence": number_between_0_and_1
        }}
    ],
    "transitions": [
        {{
            "from_event_id": "evt_1",
            "to_event_id": "evt_2",
            "time_elapsed": "string or null",
            "transition_type": "sequential|concurrent|overlapping",
            "confidence": number_between_0_and_1
        }}
    ],
    "summary": "string or null"
}}

Respond ONLY with valid JSON, no additional text.
"""

                payload = {
                    "model": self.model,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": max_tokens
                    }
                }
                response = requests.post(self.base_url, json=payload)
                response.raise_for_status()
                response_text = response.json()["response"]

                # Clean response
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                # Parse and validate with Pydantic
                response_data = json.loads(response_text)
                return PatientTimeline(**response_data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {response_text}")
            return PatientTimeline(document_id="", events=[], transitions=[])

        except Exception as e:
            logger.error(f"Error in generate_structured_timeline: {e}")
            return PatientTimeline(document_id="", events=[], transitions=[])

        return PatientTimeline(document_id="", events=[], transitions=[])


class TimelineExtractor:
    """Main class for extracting event timelines from medical texts"""

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = None,
        cache_dir: str = "rag_llm_approach/cache_timelines",
        output_file: str = "patient_timelines.json",
        checkpoint_file: str = "rag_llm_approach/timeline_extraction_checkpoint.json"
    ):
        self.llm = LLMInterface(provider=llm_provider, model=model)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.output_file = Path(output_file)
        self.checkpoint_file = Path(checkpoint_file)

        # Track processed documents
        self.processed_docs: Set[str] = set()
        self.timelines: List[Dict] = []

        # Load existing results and checkpoint
        self._load_checkpoint()
        self._load_existing_timelines()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load result from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def _load_checkpoint(self):
        """Load checkpoint to resume processing"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.processed_docs = set(checkpoint.get("processed_docs", []))
                    logger.info(f"Loaded checkpoint: {len(self.processed_docs)} documents already processed")
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
                self.processed_docs = set()

    def _save_checkpoint(self):
        """Save checkpoint for resumable processing"""
        checkpoint = {
            "processed_docs": list(self.processed_docs),
            "last_updated": datetime.now().isoformat(),
            "total_processed": len(self.processed_docs)
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Checkpoint saved: {len(self.processed_docs)} documents")
        except Exception as e:
            logger.warning(f"Error saving checkpoint: {e}")

    def _load_existing_timelines(self):
        """Load existing timelines from output file"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    self.timelines = data.get("timelines", [])
                    logger.info(f"Loaded {len(self.timelines)} existing timelines")
            except Exception as e:
                logger.warning(f"Error loading existing timelines: {e}")
                self.timelines = []

    def _construct_timeline_sequence(self, timeline: PatientTimeline) -> List:
        """
        Construct a sequential timeline representation where concurrent events are grouped.

        Args:
            timeline: PatientTimeline object with events

        Returns:
            List representation like ['E1', ['E2', 'E3'], 'E4', 'E5'] where sublists
            represent concurrent events
        """
        if not timeline.events:
            return []

        # Sort events by sequence position
        sorted_events = sorted(timeline.events, key=lambda e: e.sequence_position)

        # Build a dictionary of concurrent event groups
        concurrent_groups = {}
        for event in sorted_events:
            event_id = event.event_id
            if event.concurrent_with:
                # Find if this event is already in a group
                found_group = None
                for group_key in concurrent_groups:
                    if event_id in concurrent_groups[group_key] or any(
                        concurrent_id in concurrent_groups[group_key]
                        for concurrent_id in event.concurrent_with
                    ):
                        found_group = group_key
                        break

                if found_group:
                    # Add to existing group
                    concurrent_groups[found_group].add(event_id)
                    concurrent_groups[found_group].update(event.concurrent_with)
                else:
                    # Create new group
                    group = {event_id}
                    group.update(event.concurrent_with)
                    concurrent_groups[event.sequence_position] = group

        # Build the timeline sequence
        sequence = []
        processed_events = set()

        for event in sorted_events:
            event_id = event.event_id

            # Skip if already processed as part of a concurrent group
            if event_id in processed_events:
                continue

            # Check if this event is part of a concurrent group
            in_group = False
            for group_key, group in concurrent_groups.items():
                if event_id in group:
                    # Add all events in this group as a list
                    group_events = sorted(list(group))
                    if len(group_events) > 1:
                        sequence.append(group_events)
                    else:
                        sequence.append(group_events[0])
                    processed_events.update(group_events)
                    in_group = True
                    break

            # If not in a group, add as single event
            if not in_group:
                sequence.append(event_id)
                processed_events.add(event_id)

        return sequence

    def _save_timelines(self):
        """Save all timelines to output file"""
        output_data = {
            "timelines": self.timelines,
            "metadata": {
                "total_timelines": len(self.timelines),
                "last_updated": datetime.now().isoformat(),
                "documents_processed": len(self.processed_docs)
            }
        }
        try:
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Timelines saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving timelines: {e}")

    def create_timeline_extraction_prompt(self, text: str, doc_id: str) -> str:
        """Create prompt for timeline extraction"""
        prompt = f"""
You are analyzing a medical discharge summary with pre-marked events (using XML tags like <event1>...</event1>, <event2>...</event2>, etc.).

Your task is to extract a complete TIMELINE of what happened to the patient. Create a chronological sequence of events showing:
1. The ORDER in which events occurred
2. Events that happened at the SAME TIME (concurrent events) - mark these with concurrent_with field
3. TIME INTERVALS between events when mentioned (e.g., "2 hours later", "the next day")
4. DURATION of events when mentioned (e.g., "surgery lasted 45 minutes", "stayed for 3 days")

IMPORTANT: When events occur simultaneously or at the same time:
- List the event IDs of concurrent events in the "concurrent_with" field
- For example, if E2 and E3 happen at the same time, E2's concurrent_with should include "E3" and E3's concurrent_with should include "E2"
- This allows reconstruction of a timeline like: [E1, [E2, E3], E4] where [E2, E3] represents concurrent events

For each event in the timeline:
- Assign a unique event_id (e.g., "evt_1", "evt_2", etc.)
- Extract the event text from the XML tags
- Determine its position in the sequence (0 for first event, 1 for second, etc.)
- Identify any concurrent events (list their IDs in concurrent_with field)
- Extract absolute time if available (e.g., "2024-01-15 14:30")
- Extract duration if mentioned
- Classify the event type

For transitions between events:
- Identify the time that elapsed between sequential events
- Mark whether events are sequential, concurrent, or overlapping
- Provide confidence scores

Medical text with pre-marked events (document ID: {doc_id}):
{text}

Focus on creating a clear, chronological timeline that represents the patient's journey through their medical care.
Extract information only from the pre-marked events in the XML tags.
"""
        return prompt

    def extract_timeline(self, document: DocumentSummary) -> Optional[PatientTimeline]:
        """Extract timeline from a document"""
        # Check if already processed
        if document.doc_id in self.processed_docs:
            logger.info(f"Document {document.doc_id} already processed, skipping")
            return None

        logger.info(f"Processing document: {document.doc_id}")

        # Check cache first
        cache_key = self._get_cache_key(document.text)
        cached_result = self._load_from_cache(cache_key)

        if cached_result:
            logger.info("Using cached timeline")
            timeline = PatientTimeline(**cached_result)
        else:
            # Generate prompt and get structured LLM response
            prompt = self.create_timeline_extraction_prompt(document.text, document.doc_id)

            try:
                timeline = self.llm.generate_structured_timeline(prompt)
                timeline.document_id = document.doc_id

                # Construct timeline sequence and add to cache
                timeline_sequence = self._construct_timeline_sequence(timeline)

                # Cache the result
                cache_data = timeline.model_dump()
                cache_data["timeline_sequence"] = timeline_sequence
                cache_data["extracted_at"] = datetime.now().isoformat()
                self._save_to_cache(cache_key, cache_data)

                logger.info(f"Extracted timeline with {len(timeline.events)} events and {len(timeline.transitions)} transitions")

            except Exception as e:
                logger.error(f"Error extracting timeline: {e}")
                return None

        # Construct timeline_sequence for both new and cached results
        timeline_sequence = self._construct_timeline_sequence(timeline)

        # Add to results with timeline_sequence included
        timeline_data = timeline.model_dump()
        timeline_data["timeline_sequence"] = timeline_sequence
        self.timelines.append(timeline_data)
        self.processed_docs.add(document.doc_id)

        # Save progress immediately (checkpoint and full output)
        self._save_checkpoint()
        self._save_timelines()

        return timeline

    def process_documents(self, documents: List[DocumentSummary]) -> List[PatientTimeline]:
        """Process multiple documents and extract timelines"""
        extracted_timelines = []
        total = len(documents)

        # Filter out already processed documents
        documents_to_process = [doc for doc in documents if doc.doc_id not in self.processed_docs]

        logger.info(f"Processing {len(documents_to_process)} documents ({total - len(documents_to_process)} already processed)")

        for i, doc in enumerate(documents_to_process, 1):
            try:
                logger.info(f"Processing document {i}/{len(documents_to_process)}: {doc.doc_id}")
                timeline = self.extract_timeline(doc)
                if timeline:
                    extracted_timelines.append(timeline)

            except KeyboardInterrupt:
                logger.warning("Processing interrupted by user. Progress has been saved.")
                break
            except Exception as e:
                logger.error(f"Error processing document {doc.doc_id}: {e}")
                continue

        # Final save
        self._save_timelines()
        logger.info(f"Completed processing. Total timelines: {len(self.timelines)}")

        return extracted_timelines

    def get_statistics(self) -> Dict:
        """Get statistics about extracted timelines"""
        if not self.timelines:
            return {"error": "No timelines extracted"}

        stats = {
            "total_timelines": len(self.timelines),
            "total_events": sum(len(t["events"]) for t in self.timelines),
            "total_transitions": sum(len(t["transitions"]) for t in self.timelines),
            "avg_events_per_timeline": sum(len(t["events"]) for t in self.timelines) / len(self.timelines),
            "avg_transitions_per_timeline": sum(len(t["transitions"]) for t in self.timelines) / len(self.timelines),
            "event_types": {},
            "concurrent_events_count": 0,
            "timed_transitions_count": 0
        }

        for timeline in self.timelines:
            for event in timeline["events"]:
                event_type = event["event_type"]
                stats["event_types"][event_type] = stats["event_types"].get(event_type, 0) + 1
                if event.get("concurrent_with"):
                    stats["concurrent_events_count"] += 1

            for transition in timeline["transitions"]:
                if transition.get("time_elapsed"):
                    stats["timed_transitions_count"] += 1

        return stats


def load_thyme_data() -> List[DocumentSummary]:
    """Load THYME dataset discharge summaries with XML-tagged events"""
    import sys
    from pathlib import Path

    data_loaders_path = Path(__file__).parent.parent / "data_loaders"
    sys.path.insert(0, str(data_loaders_path))

    try:
        import thyme_loader
        return thyme_loader.documents
    except Exception as e:
        logger.warning(f"Could not load from thyme_loader: {e}")
        # Fallback to basic loading
        import cryptpandas as crp
        df = crp.read_encrypted(path='thyme.crypt', password=os.environ['TP'])
        documents = [
            DocumentSummary(doc_id=row["document_id"], text=row["text"])
            for _, row in df.drop_duplicates(subset="document_id")[["document_id", "text"]].iterrows()
        ]
        return documents


def main():
    """Main function to run the timeline extraction pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract event timelines from medical texts")
    parser.add_argument("--provider", default="ollama", choices=["openai", "ollama"],
                        help="LLM provider to use")
    parser.add_argument("--model", default="gemma3:27b",
                        help="Model name (default: gemma3:27b for ollama, gpt-4 for openai)")
    parser.add_argument("--output", default="rag_llm_approach/patient_timelines.json",
                        help="Output file for timelines")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of documents to process (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")

    args = parser.parse_args()

    print("=" * 70)
    print("Event Timeline Extraction from Medical Discharge Summaries")
    print("=" * 70)
    print(f"LLM Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Output File: {args.output}")
    print(f"Resume Mode: {args.resume}")
    print("=" * 70)

    # Initialize extractor
    extractor = TimelineExtractor(
        llm_provider=args.provider,
        model=args.model,
        output_file=args.output
    )

    # Load documents
    documents = load_thyme_data()
    print(f"\nLoaded {len(documents)} documents from dataset")

    # Apply limit if specified
    if args.limit:
        documents = documents[:args.limit]
        print(f"Limited to first {args.limit} documents for testing")

    # Process documents
    print(f"\nStarting timeline extraction...")
    print(f"(Press Ctrl+C to stop - progress will be saved automatically)\n")

    timelines = extractor.process_documents(documents)

    # Display statistics
    stats = extractor.get_statistics()
    print("\n" + "=" * 70)
    print("TIMELINE EXTRACTION STATISTICS")
    print("=" * 70)
    print(f"Total timelines extracted: {stats['total_timelines']}")
    print(f"Total events: {stats['total_events']}")
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"Average events per timeline: {stats['avg_events_per_timeline']:.2f}")
    print(f"Average transitions per timeline: {stats['avg_transitions_per_timeline']:.2f}")
    print(f"Events with concurrent occurrences: {stats['concurrent_events_count']}")
    print(f"Transitions with time information: {stats['timed_transitions_count']}")

    print(f"\nEvent types distribution:")
    for event_type, count in stats['event_types'].items():
        print(f"  {event_type}: {count}")

    # Show sample timeline
    if timelines:
        print("\n" + "=" * 70)
        print("SAMPLE TIMELINE")
        print("=" * 70)
        sample = timelines[0]
        print(f"Document: {sample.document_id}")
        if sample.summary:
            print(f"Summary: {sample.summary}")

        # Display timeline sequence from stored data
        if extractor.timelines:
            sample_data = extractor.timelines[0]
            if sample_data.get("timeline_sequence"):
                print(f"\nTimeline Sequence:")
                print(f"  {sample_data['timeline_sequence']}")
                print(f"  (Note: sublists represent concurrent events)")

        print(f"\nEvents ({len(sample.events)}):")
        for event in sample.events[:5]:  # Show first 5 events
            print(f"  [{event.sequence_position}] {event.event_text}")
            print(f"      Type: {event.event_type}, Confidence: {event.confidence:.2f}")
            if event.temporal_expression:
                print(f"      Temporal: {event.temporal_expression}")
            if event.absolute_time:
                print(f"      Absolute time: {event.absolute_time}")
            if event.duration:
                print(f"      Duration: {event.duration}")
            if event.concurrent_with:
                print(f"      Concurrent with: {', '.join(event.concurrent_with)}")

        if len(sample.events) > 5:
            print(f"  ... and {len(sample.events) - 5} more events")

        print(f"\nTransitions ({len(sample.transitions)}):")
        for trans in sample.transitions[:5]:  # Show first 5 transitions
            print(f"  {trans.from_event_id} → {trans.to_event_id}")
            print(f"      Type: {trans.transition_type}, Confidence: {trans.confidence:.2f}")
            if trans.time_elapsed:
                print(f"      Time elapsed: {trans.time_elapsed}")

    print("\n" + "=" * 70)
    print(f"Timelines saved to: {extractor.output_file}")
    print(f"Checkpoint saved to: {extractor.checkpoint_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

