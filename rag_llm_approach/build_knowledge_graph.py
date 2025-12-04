#!/usr/bin/env python3
"""
Temporal Event Extraction from Medical Discharge Summaries

This program extracts events and their temporal information from medical discharge summaries
using either OpenAI GPT-4 or local Ollama Gemma model. It focuses on events that explicitly
mention their time of occurrence or duration in the text.
"""

import json
import os
import cryptpandas as crp
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Third-party imports
import openai
import requests
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class EventType(str, Enum):
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"
    DIAGNOSIS = "diagnosis"
    TEST = "test"
    OTHER = "other"

class TemporalType(str, Enum):
    ABSOLUTE_TIME = "absolute_time"
    DURATION = "duration"
    RELATIVE_TIME = "relative_time"

class RelationType(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"

class ExtractedEvent(BaseModel):
    event_text: str = Field(description="Exact event description from the text")
    event_type: EventType = Field(description="Type of medical event")
    temporal_expression: str = Field(description="Exact temporal phrase from text")
    temporal_type: TemporalType = Field(description="Type of temporal information")
    normalized_time: Optional[str] = Field(None, description="Standardized format if possible")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    sentence_context: str = Field(description="Full sentence containing the event")

class ExtractedRelation(BaseModel):
    event1_text: str = Field(description="First event description")
    event2_text: str = Field(description="Second event description")
    relation_type: RelationType = Field(description="Type of temporal relationship")
    time_difference: Optional[str] = Field(None, description="Time difference if determinable")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

class TemporalExtractionResponse(BaseModel):
    events: List[ExtractedEvent] = Field(description="List of extracted temporal events")
    relations: List[ExtractedRelation] = Field(description="List of temporal relations between events")

@dataclass
class TemporalEvent:
    """Represents an event with temporal information"""
    event_text: str
    event_type: str  # e.g., "medication", "procedure", "symptom", "diagnosis"
    temporal_expression: str  # The exact temporal phrase from text
    temporal_type: str  # "absolute_time", "duration", "relative_time"
    normalized_time: Optional[str] = None  # Standardized representation
    confidence: float = 0.0  # LLM confidence score
    document_id: str = ""
    sentence_context: str = ""

@dataclass
class EventRelation:
    """Represents temporal relationship between two events"""
    event1_id: str
    event2_id: str
    relation_type: str  # "before", "after", "during", "overlaps"
    time_difference: Optional[str] = None  # e.g., "2 days", "3 hours"
    confidence: float = 0.0

@dataclass
class DocumentSummary:
    """Sample medical discharge summary for testing"""
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

    def generate_structured_response(self, prompt: str, max_tokens: int = 2000) -> TemporalExtractionResponse:
        """Generate structured response using LLM with proper schema validation"""
        response_text = ""  # Initialize to avoid reference errors

        try:
            if self.provider == "openai":
                # Use OpenAI's structured output with schema validation
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a medical NLP expert that extracts temporal events from medical texts. "
                                "Your task is to identify medical events and their temporal relationships from clinical documents. "
                                "Extract events with their temporal expressions and determine relationships between events. "
                                "Be precise and only extract events that have clear temporal information. "
                                "Provide confidence scores based on the clarity of temporal expressions."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    text_format=TemporalExtractionResponse
                )
                return response.output_parsed

            elif self.provider == "ollama":
                # For Ollama, we'll use the prompt with clear JSON schema instructions
                enhanced_prompt = f"""
{prompt}

IMPORTANT: You must respond with valid JSON that exactly matches this schema:
{{
    "events": [
        {{
            "event_text": "string",
            "event_type": "medication|procedure|symptom|diagnosis|test|other",
            "temporal_expression": "string",
            "temporal_type": "absolute_time|duration|relative_time",
            "normalized_time": "string or null",
            "confidence": number_between_0_and_1,
            "sentence_context": "string"
        }}
    ],
    "relations": [
        {{
            "event1_text": "string",
            "event2_text": "string", 
            "relation_type": "before|after|during|overlaps",
            "time_difference": "string or null",
            "confidence": number_between_0_and_1
        }}
    ]
}}
Respond only with valid JSON, no additional text.
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

                # Clean the response text to extract JSON
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                # Parse and validate with Pydantic
                response_data = json.loads(response_text)
                return TemporalExtractionResponse(**response_data)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response_text}")
            return TemporalExtractionResponse(events=[], relations=[])

        except Exception as e:
            print(f"Error in generate_structured_response: {e}")
            return TemporalExtractionResponse(events=[], relations=[])

        return TemporalExtractionResponse(events=[], relations=[])

    def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Legacy method - kept for backward compatibility"""
        structured_response = self.generate_structured_response(prompt, max_tokens)
        return structured_response.model_dump_json(indent=2)


class TemporalEventExtractor:
    """Main class for extracting temporal events from medical texts"""

    def __init__(self, llm_provider: str = "openai", model: str = None, cache_dir: str = "rag_llm_approach/cache_temporal"):
        self.llm = LLMInterface(provider=llm_provider, model=model)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Temporal expression patterns for validation
        self.temporal_patterns = [
            r'\b\d{1,2}:\d{2}\b',  # Time like 14:30
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Date like 12/25/2023
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b',  # Duration
            r'\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|this\s+\w+)\b',  # Relative time
            r'\b(?:before|after|during|while|when|since|until|for)\s+\w+',  # Temporal prepositions
        ]

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

    def create_extraction_prompt(self, text: str) -> str:
        """Create prompt for temporal event extraction"""
        prompt = f"""
You are a medical NLP expert. Analyze the following medical discharge summary that has events pre-marked with XML tags (e.g., <event1>...</event1>, <event2>...</event2>, etc.).

Your task is to:
1. Focus on the events that are already marked with XML tags in the text
2. Extract temporal information for these pre-marked events
3. Identify temporal relationships between the marked events

For each marked event, provide:
1. The exact event description from within the XML tags
2. Event type (Event type must be one of: medication, procedure, symptom, diagnosis, test, other. If unsure, use "other".)
3. The exact temporal expression associated with the event (if any)
4. Temporal type: "absolute_time" (specific date/time), "duration" (how long), or "relative_time" (relative to something)
5. Confidence score (0.0-1.0)
6. The sentence containing the event

For temporal relationships:
- Look for clear temporal connections between the marked events
- Also extract events if there is a clear temporal relation between two events, either the amount of time that has passed or a relation like before/after
- Identify relationships like before, after, during, overlaps
- Determine time differences when possible

Medical text with pre-marked events:
{text}

Respond in valid JSON format with this structure:
{{
    "events": [
        {{
            "event_text": "exact event description from XML tags",
            "event_type": "medication|procedure|symptom|diagnosis|test|other",
            "temporal_expression": "exact temporal phrase from text",
            "temporal_type": "absolute_time|duration|relative_time",
            "normalized_time": "standardized format if possible",
            "confidence": 0.95,
            "sentence_context": "full sentence containing the event"
        }}
    ],
    "relations": [
        {{
            "event1_text": "first event from XML tags",
            "event2_text": "second event from XML tags",
            "relation_type": "before|after|during|overlaps",
            "time_difference": "2 days|3 hours|etc if determinable",
            "confidence": 0.85
        }}
    ]
}}

Focus on the pre-marked events and their temporal information. Be precise and use the XML markup to guide your extraction.
"""
        return prompt

    def extract_temporal_events(self, document: DocumentSummary) -> Tuple[List[TemporalEvent], List[EventRelation]]:
        """Extract temporal events from a document using structured output"""
        logger.info(f"Processing document: {document.doc_id}")

        # Check cache first
        cache_key = self._get_cache_key(document.text)
        cached_result = self._load_from_cache(cache_key)

        if cached_result:
            logger.info("Using cached result")
            events = [TemporalEvent(**event) for event in cached_result.get("events", [])]
            relations = [EventRelation(**rel) for rel in cached_result.get("relations", [])]
            return events, relations

        # Generate prompt and get structured LLM response
        prompt = self.create_extraction_prompt(document.text)

        try:
            # Use structured output with Pydantic validation
            structured_response = self.llm.generate_structured_response(prompt)

            # Convert Pydantic models to dataclass objects
            events = []
            for extracted_event in structured_response.events:
                event = TemporalEvent(
                    event_text=extracted_event.event_text,
                    event_type=extracted_event.event_type.value,
                    temporal_expression=extracted_event.temporal_expression,
                    temporal_type=extracted_event.temporal_type.value,
                    normalized_time=extracted_event.normalized_time,
                    confidence=extracted_event.confidence,
                    document_id=document.doc_id,
                    sentence_context=extracted_event.sentence_context
                )
                events.append(event)

            # Convert relations
            relations = []
            for extracted_relation in structured_response.relations:
                relation = EventRelation(
                    event1_id=extracted_relation.event1_text,
                    event2_id=extracted_relation.event2_text,
                    relation_type=extracted_relation.relation_type.value,
                    time_difference=extracted_relation.time_difference,
                    confidence=extracted_relation.confidence
                )
                relations.append(relation)

            # Cache the result
            cache_data = {
                "events": [asdict(event) for event in events],
                "relations": [asdict(relation) for relation in relations],
                "timestamp": datetime.now().isoformat()
            }
            self._save_to_cache(cache_key, cache_data)

            logger.info(f"Extracted {len(events)} events and {len(relations)} relations")
            return events, relations

        except Exception as e:
            logger.error(f"Error processing extraction with structured output: {e}")
            return [], []


def load_thyme_data() -> List[DocumentSummary]:
    """
    Load THYME dataset discharge summaries with XML-tagged events.
    Uses the thyme_loader to extract events and create tagged versions.
    """
    # Import the thyme_loader module to get the processed documents
    import sys
    from pathlib import Path

    # Add the data_loaders directory to Python path
    data_loaders_path = Path(__file__).parent.parent / "data_loaders"
    sys.path.insert(0, str(data_loaders_path))

    try:
        import thyme_loader
        # The thyme_loader module creates the documents list when imported
        return thyme_loader.documents
    except Exception as e:
        logger.warning(f"Could not load from thyme_loader: {e}. Falling back to basic loading.")
        # Fallback to basic loading if thyme_loader fails
        df = crp.read_encrypted(path='thyme.crypt', password=os.environ['TP'])
        documents = [
            DocumentSummary(doc_id=row["document_id"], text=row["text"])
            for _, row in df.drop_duplicates(subset="document_id")[["document_id", "text"]].iterrows()
        ]
        return documents

def load_sample_data() -> List[DocumentSummary]:
    """
    Load sample medical discharge summaries for testing.
    This function should be replaced with actual data loading logic.
    """
    sample_documents = [
        DocumentSummary(
            doc_id="sample_001",
            text="""
            Patient was admitted on January 15, 2024 at 14:30 with chest pain. 
            Blood tests were performed 2 hours after admission showing elevated troponin levels. 
            The patient received aspirin 325mg at 16:45 and was started on heparin infusion. 
            Cardiac catheterization was performed the following morning at 08:00 on January 16, 2024, 
            revealing 90% stenosis of the LAD. PCI was completed successfully after 45 minutes. 
            The patient remained stable for 3 days post-procedure and was discharged on January 19, 2024 
            with instructions to take clopidogrel 75mg daily for 12 months.
            """
        ),
        DocumentSummary(
            doc_id="sample_002",
            text="""
            The patient presented to the emergency department on March 3, 2024 at 22:15 
            complaining of severe abdominal pain that started 6 hours earlier. 
            CT scan performed at 23:30 showed acute appendicitis. 
            Laparoscopic appendectomy was performed at 02:00 on March 4, 2024. 
            Surgery lasted 30 minutes without complications. 
            Patient recovered well and was discharged 24 hours post-operatively 
            on March 5, 2024 with oral antibiotics for 7 days.
            """
        ),
        DocumentSummary(
            doc_id="sample_003",
            text="""
            Elderly patient with history of diabetes mellitus type 2 for 15 years 
            was admitted on February 10, 2024 for diabetic ketoacidosis. 
            Initial glucose level was 450 mg/dL at admission time 09:20. 
            IV insulin therapy was initiated immediately and continued for 48 hours. 
            Patient's condition improved gradually over the next 2 days. 
            Blood glucose normalized by February 13, 2024. 
            Patient was transitioned to subcutaneous insulin and discharged 
            after 5 days of hospitalization with diabetes education.
            """
        )
    ]

    return sample_documents


class KnowledgeBase:
    """Manages the temporal event knowledge base"""

    def __init__(self, kb_file: str = "temporal_events_kb.json"):
        self.kb_file = Path(kb_file)
        self.events = []
        self.relations = []
        self.statistics = {}

    def add_events(self, events: List[TemporalEvent], relations: List[EventRelation]):
        """Add events and relations to the knowledge base"""
        self.events.extend(events)
        self.relations.extend(relations)

    def compute_statistics(self) -> Dict:
        """Compute statistics about temporal events"""
        stats = {
            "total_events": len(self.events),
            "total_relations": len(self.relations),
            "event_types": {},
            "temporal_types": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "documents_processed": len(set(event.document_id for event in self.events))
        }

        for event in self.events:
            # Count event types
            stats["event_types"][event.event_type] = stats["event_types"].get(event.event_type, 0) + 1

            # Count temporal types
            stats["temporal_types"][event.temporal_type] = stats["temporal_types"].get(event.temporal_type, 0) + 1

            # Confidence distribution
            if event.confidence >= 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif event.confidence >= 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1

        self.statistics = stats
        return stats

    def save_to_file(self):
        """Save knowledge base to JSON file"""
        data = {
            "events": [asdict(event) for event in self.events],
            "relations": [asdict(relation) for relation in self.relations],
            "statistics": self.statistics,
            "generated_at": datetime.now().isoformat()
        }

        with open(self.kb_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Knowledge base saved to {self.kb_file}")

    def load_from_file(self):
        """Load knowledge base from JSON file"""
        if self.kb_file.exists():
            with open(self.kb_file, 'r') as f:
                data = json.load(f)

            self.events = [TemporalEvent(**event) for event in data.get("events", [])]
            self.relations = [EventRelation(**rel) for rel in data.get("relations", [])]
            self.statistics = data.get("statistics", {})

            logger.info(f"Knowledge base loaded from {self.kb_file}")


def main():
    """Main function to run the temporal event extraction pipeline"""

    # Configuration
    # LLM_PROVIDER = "openai"  # Change to "ollama" for local Gemma
    LLM_PROVIDER = "ollama"
    # MODEL = "gpt-4.1"
    MODEL = "gemma3:27b"

    print("Starting Temporal Event Extraction from Medical Discharge Summaries")
    print(f"Using LLM: {LLM_PROVIDER} - {MODEL}")

    # Initialize components
    extractor = TemporalEventExtractor(llm_provider=LLM_PROVIDER, model=MODEL)
    knowledge_base = KnowledgeBase()

    # Load existing knowledge base if available
    knowledge_base.load_from_file()

    documents = load_thyme_data()
    print(f"Loaded {len(documents)} documents")

    # For testing purposes, only process first 5 documents
    documents = documents[:5]
    print(f"Processing only first {len(documents)} documents for testing")

    # Process each document
    all_events = []
    all_relations = []

    for doc in documents:
        try:
            events, relations = extractor.extract_temporal_events(doc)
            all_events.extend(events)
            all_relations.extend(relations)

            print(f"Document {doc.doc_id}: {len(events)} events, {len(relations)} relations")

        except Exception as e:
            logger.error(f"Error processing document {doc.doc_id}: {e}")
            continue

    # Add to knowledge base
    knowledge_base.add_events(all_events, all_relations)

    # Compute and display statistics
    stats = knowledge_base.compute_statistics()
    print("\n=== TEMPORAL EVENT EXTRACTION STATISTICS ===")
    print(f"Total events extracted: {stats['total_events']}")
    print(f"Total temporal relations: {stats['total_relations']}")
    print(f"Documents processed: {stats['documents_processed']}")

    print(f"\nEvent types distribution:")
    for event_type, count in stats['event_types'].items():
        print(f"  {event_type}: {count}")

    print(f"\nTemporal types distribution:")
    for temp_type, count in stats['temporal_types'].items():
        print(f"  {temp_type}: {count}")

    print(f"\nConfidence distribution:")
    for conf_level, count in stats['confidence_distribution'].items():
        print(f"  {conf_level} (>0.8, 0.5-0.8, <0.5): {count}")

    # Save knowledge base
    knowledge_base.save_to_file()
    print(f"\nKnowledge base saved to {knowledge_base.kb_file}")

    # Show some sample events
    print(f"\n=== SAMPLE EXTRACTED EVENTS ===")
    for i, event in enumerate(all_events[:5]):  # Show first 5 events
        print(f"\nEvent {i+1}:")
        print(f"  Text: {event.event_text}")
        print(f"  Type: {event.event_type}")
        print(f"  Temporal: {event.temporal_expression} ({event.temporal_type})")
        print(f"  Confidence: {event.confidence:.2f}")


if __name__ == "__main__":
    main()
