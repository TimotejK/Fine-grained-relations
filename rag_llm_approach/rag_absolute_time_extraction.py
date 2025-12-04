import json
import pandas as pd
import torch
import argparse
from datetime import datetime
from difflib import get_close_matches
from unsloth import FastLanguageModel
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import compute_metrics, store_prediction_for_error_analysis
from evaluation.error_analysis import save_for_error_analysis

class RAGTemporalExtractor:
    def __init__(self, model_path="unsloth/gemma-3-12b-it-unsloth-bnb-4bit", use_finetuned=False):
        self.model_path = model_path
        self.use_finetuned = use_finetuned
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the Gemma 3 12B model using unsloth for efficient inference"""
        try:
            if self.use_finetuned:
                # Load fine-tuned model
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name="./rag_finetuned_model",
                    max_seq_length=4096,
                    dtype=None,
                    load_in_4bit=True,
                )
            else:
                # Load base model
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=4096,
                    dtype=None,
                    load_in_4bit=True,
                )

            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            print(f"Model loaded successfully: {self.model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def call_llm(self, prompt):
        """Call the local Gemma 3 12B model for inference"""
        try:
            # Format prompt for Gemma chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate
            inputs = self.tokenizer(
                text=formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Parse the response to extract temporal information
            return self.parse_temporal_response(response)

        except Exception as e:
            print(f"Error in LLM call: {e}")
            return {"start": None, "end": None, "error": str(e)}

    def parse_temporal_response(self, response):
        """Parse LLM response to extract start and end times"""
        try:
            # Try to parse as JSON first
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed

            # Fallback: extract using regex patterns
            import re
            start_pattern = r"start[\"']?\s*:\s*[\"']?([^,}\"\n]+)"
            end_pattern = r"end[\"']?\s*:\s*[\"']?([^,}\"\n]+)"

            start_match = re.search(start_pattern, response, re.IGNORECASE)
            end_match = re.search(end_pattern, response, re.IGNORECASE)

            return {
                "start": start_match.group(1).strip() if start_match else None,
                "end": end_match.group(1).strip() if end_match else None
            }

        except Exception as e:
            return {"start": None, "end": None, "parse_error": str(e)}

# Global instance for backward compatibility
rag_extractor = None

def call_llm(prompt):
    """Backward compatibility function"""
    global rag_extractor
    if rag_extractor is None:
        rag_extractor = RAGTemporalExtractor()
    return rag_extractor.call_llm(prompt)

# Load the knowledge base
def load_event_kb(kb_path):
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    return kb['events']

# Retrieve relevant KB entries for an event
def retrieve_relevant_kb_entries(event_text, kb_events, top_k=3):
    kb_texts = [e['event_text'] for e in kb_events]
    matches = get_close_matches(event_text, kb_texts, n=top_k, cutoff=0.5)
    return [e for e in kb_events if e['event_text'] in matches]

# Construct a RAG prompt for the LLM
def construct_rag_prompt(event_row, kb_entries):
    prompt = (
        f"You are a clinical temporal reasoning expert. Your task is to predict the start and end times of medical events.\n\n"
        f"Event: {event_row.get('event_text', event_row.get('event', ''))}\n"
        f"Context: {event_row.get('context', '')}\n\n"
        f"Relevant knowledge base entries about similar events:\n"
    )
    for kb in kb_entries:
        prompt += (
            f"- Event: {kb['event_text']} | Type: {kb['event_type']} | Time: {kb['temporal_expression']} | "
            f"Normalized: {kb['normalized_time']} | Context: {kb['sentence_context']}\n"
        )
    prompt += (
        "\n\nTask: Predict the absolute start and end times for this event. You MUST provide specific temporal predictions.\n\n"
        "Guidelines:\n"
        "1. If the time is explicitly mentioned in the context, use that information.\n"
        "2. If the time is NOT explicitly mentioned, infer the most likely time based on:\n"
        "   - The event type (e.g., surgeries typically happen during business hours, symptoms may occur at any time)\n"
        "   - Similar events in the knowledge base\n"
        "   - Clinical context and typical medical practice patterns\n"
        "   - Temporal relations to other events mentioned in the context\n"
        "3. Consider typical durations for different event types when predicting start and end times.\n\n"
        "Return your answer as a JSON object with 'start' and 'end' fields in ISO format (YYYY-MM-DDTHH:MM:SS).\n"
        "Example: {\"start\": \"1900-01-15T09:30:00\", \"end\": \"1900-01-15T11:00:00\"}\n\n"
        "You must provide concrete temporal predictions even if the exact time is not stated in the text."
    )
    return prompt


def generate_ground_truth_response(row):
    """Generate ground truth response for fine-tuning based on actual temporal data"""
    try:
        # Convert minutes to ISO format for response
        base_date = datetime(1900, 1, 1, 0, 0, 0)

        start_time = base_date + pd.Timedelta(minutes=row['start_time_minutes'])
        end_time = base_date + pd.Timedelta(minutes=row['end_time_minutes'])

        response = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }

        return json.dumps(response)

    except Exception as e:
        return json.dumps({"start": "1900-01-01T00:00:00", "end": "1900-01-01T00:00:00"})

def generate_finetuning_dataset():
    """Generate QA pairs for fine-tuning the RAG model"""
    print("Generating fine-tuning dataset...")

    # Load training data
    df = load_i2b2_absolute_data(test_split=False)

    # Load event knowledge base
    try:
        kb_events = load_event_kb('rag_llm_approach/temporal_events_kb.json')
    except FileNotFoundError:
        print("Warning: Knowledge base not found, using empty KB")
        kb_events = []

    qa_pairs = []
    for idx, row in df.iterrows():
        try:
            # Extract event string from text using start_char and end_char
            event_string = row['text'][row['start_char']:row['end_char']]
            kb_entries = retrieve_relevant_kb_entries(event_string, kb_events)

            # Generate prompt
            prompt = construct_rag_prompt(
                {'event_text': event_string, 'context': row['text']},
                kb_entries
            )

            # Generate ground truth response
            ground_truth = generate_ground_truth_response(row)

            qa_pairs.append({
                'instruction': 'Extract temporal information for clinical events using provided knowledge base.',
                'input': prompt,
                'output': ground_truth,
                'metadata': {
                    'document_id': row['document_id'],
                    'event_text': event_string,
                    'start_char': row['start_char'],
                    'end_char': row['end_char']
                }
            })

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} examples")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Save dataset
    output_path = 'rag_llm_approach/rag_finetuning_dataset.json'
    with open(output_path, 'w') as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Generated {len(qa_pairs)} QA pairs for fine-tuning")
    print(f"Dataset saved to: {output_path}")

    return qa_pairs

def transform_rag_prediction_to_evaluation_format(llm_response, admission_time):
    """Transform RAG LLM response to evaluation format similar to evaluate_llm_prompting"""
    try:
        # Parse ISO timestamps to minutes if available
        if llm_response.get("start") and llm_response.get("end"):
            try:
                start_dt = datetime.fromisoformat(llm_response["start"])
                end_dt = datetime.fromisoformat(llm_response["end"])
                
                # Convert to minutes from base date (1900-01-01)
                base_date = datetime(1900, 1, 1, 0, 0, 0)
                start_minutes = int((start_dt - base_date).total_seconds() / 60)
                end_minutes = int((end_dt - base_date).total_seconds() / 60)
            except (ValueError, TypeError):
                # Fallback to admission time if parsing fails
                start_minutes = admission_time
                end_minutes = admission_time
        else:
            start_minutes = admission_time
            end_minutes = admission_time
        
        # Create bounds (±60 minutes tolerance)
        start_lower = start_minutes - 60
        start_upper = start_minutes + 60
        end_lower = end_minutes - 60
        end_upper = end_minutes + 60
        
        duration_minutes = end_minutes - start_minutes
        duration_lower = duration_minutes - 60
        duration_upper = duration_minutes + 60
        
        prediction = ((start_minutes, start_lower, start_upper),
                     (end_minutes, end_lower, end_upper),
                     (duration_minutes, duration_lower, duration_upper))
        return prediction
        
    except Exception as e:
        print(f"Error transforming prediction: {e}")
        # Return default prediction with admission time
        prediction = ((admission_time, admission_time - 60, admission_time + 60),
                     (admission_time, admission_time - 60, admission_time + 60),
                     (60, 0, 120))
        return prediction

def transform_row_to_evaluation_format(row):
    """Transform ground truth row to evaluation format"""
    start_time = row["start_time_minutes"]
    start_lower = row["start_lower_minutes"]
    start_upper = row["start_upper_minutes"]
    end_time = row["end_time_minutes"]
    end_lower = row["end_lower_minutes"]
    end_upper = row["end_upper_minutes"]
    duration_time = row["duration_minutes"]
    duration_lower = row["duration_lower_minutes"]
    duration_upper = row["duration_upper_minutes"]

    prediction = ((start_time, start_lower, start_upper),
                  (end_time, end_lower, end_upper),
                  (duration_time, duration_lower, duration_upper))
    return prediction

def log_text(line, log_file="rag_evaluation.log"):
    """Log evaluation results to file"""
    with open(log_file, "a") as log:
        log.write(line)
        log.write("\n")

def evaluate_rag_model_performance(results, model_id="rag_model", use_finetuned=False):
    """Evaluate RAG model performance using the same metrics as other LLM evaluations"""
    print(f"Starting evaluation for {model_id}...")
    
    # Prepare evaluation data
    predictions_starts = []
    predictions_ends = []
    predictions_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []
    
    successful_predictions = 0
    total_predictions = len(results)
    
    # Load test data for ground truth bounds
    df_test = load_i2b2_absolute_data(test_split=True)
    df_test_dict = df_test.set_index(['document_id']).to_dict('index')
    
    for result in results:
        try:
            document_id = result['document_id']
            llm_response = result['llm_response']
            ground_truth = result['ground_truth']
            
            # Get full row data for bounds
            if document_id in df_test_dict:
                row_data = df_test_dict[document_id]
                admission_time = row_data.get('admission_date_minutes', 0)
            else:
                admission_time = 0
            
            # Transform prediction to evaluation format
            pred_start, pred_end, pred_duration = transform_rag_prediction_to_evaluation_format(
                llm_response, admission_time
            )
            predictions_starts.append(pred_start)
            predictions_ends.append(pred_end)
            predictions_durations.append(pred_duration)
            
            # Transform ground truth to evaluation format
            # Find matching row in test data
            matching_rows = df_test[df_test['document_id'] == document_id]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                gold_start, gold_end, gold_duration = transform_row_to_evaluation_format(row)
                gold_starts.append(gold_start)
                gold_ends.append(gold_end)
                gold_durations.append(gold_duration)
                
                # Store for error analysis
                store_prediction_for_error_analysis(
                    model_id, document_id, row["text"], 
                    row.get("event_id", 0), int(row["start_char"]), int(row["end_char"]),
                    pred_start, pred_end, pred_duration,
                    gold_start, gold_end, gold_duration
                )
                
                # Count successful predictions (has both start and end)
                if llm_response.get('start') and llm_response.get('end'):
                    successful_predictions += 1
            else:
                print(f"Warning: No matching test data found for document {document_id}")
                
        except Exception as e:
            print(f"Error processing result for document {result.get('document_id', 'unknown')}: {e}")
            continue
    
    # Compute evaluation metrics
    if predictions_starts and gold_starts:
        eval_metrics = compute_metrics(
            predictions_starts, predictions_ends, predictions_durations,
            gold_starts, gold_ends, gold_durations
        )
        
        # Add success rate to metrics
        success_rate = successful_predictions / total_predictions if total_predictions > 0 else 0
        eval_metrics['successful_predictions'] = successful_predictions
        eval_metrics['total_predictions'] = total_predictions
        eval_metrics['success_rate'] = success_rate
        
        print(f"\n{model_id} --- Final Evaluation Metrics:")
        print(f"Success Rate: {success_rate:.3f} ({successful_predictions}/{total_predictions})")
        for metric, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Log results
        log_file = f"{model_id}_finetuned_evaluation.log" if use_finetuned else f"{model_id}_evaluation.log"
        log_text(f"{model_id} --- Final Evaluation Metrics: {eval_metrics}", log_file=log_file)
        log_text(f"Success Rate: {success_rate:.3f} ({successful_predictions}/{total_predictions})", log_file=log_file)
        
        # Save detailed error analysis
        save_for_error_analysis(model_id)
        
        # Save evaluation summary
        eval_summary = {
            'model_id': model_id,
            'use_finetuned': use_finetuned,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': eval_metrics,
            'success_rate': success_rate,
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions
        }
        
        summary_file = f"rag_evaluation_summary_{'finetuned' if use_finetuned else 'base'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(eval_summary, f, indent=2)
        
        print(f"\nEvaluation summary saved to: {summary_file}")
        print(f"Detailed logs saved to: {log_file}")
        
        return eval_metrics
    else:
        print("No valid predictions found for evaluation")
        return {}

def main():
    parser = argparse.ArgumentParser(description='RAG Temporal Extraction')
    parser.add_argument('--mode', choices=['inference', 'generate_data'],
                       default='inference', help='Mode to run the system')
    parser.add_argument('--use_finetuned', action='store_true',
                       help='Use fine-tuned model for inference')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run comprehensive evaluation after inference')

    args = parser.parse_args()

    if args.mode == 'generate_data':
        generate_finetuning_dataset()
        return

    # Inference mode
    print(f"Running inference mode, use_finetuned: {args.use_finetuned}")

    # Initialize RAG extractor
    global rag_extractor
    rag_extractor = RAGTemporalExtractor(use_finetuned=args.use_finetuned)

    # Load test data
    df = load_i2b2_absolute_data(test_split=True)
    df = df[:10]

    # Load event knowledge base
    try:
        kb_events = load_event_kb('rag_llm_approach/temporal_events_kb.json')
    except FileNotFoundError:
        print("Warning: Knowledge base not found, using empty KB")
        kb_events = []

    results = []
    for idx, row in df.iterrows():
        try:
            # Extract event string from text using start_char and end_char
            event_string = row['text'][row['start_char']:row['end_char']]
            kb_entries = retrieve_relevant_kb_entries(event_string, kb_events)
            prompt = construct_rag_prompt({'event_text': event_string, 'context': row['text']}, kb_entries)

            llm_response = rag_extractor.call_llm(prompt)

            results.append({
                'document_id': row['document_id'],
                'event_text': event_string,
                'llm_response': llm_response,
                'ground_truth': {
                    'start_time_minutes': row['start_time_minutes'],
                    'end_time_minutes': row['end_time_minutes']
                }
            })

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(df)} test examples")

        except Exception as e:
            print(f"Error processing test row {idx}: {e}")
            continue

    # Save results
    output_file = 'rag_llm_approach/rag_absolute_time_predictions_finetuned.json' if args.use_finetuned else 'rag_llm_approach/rag_absolute_time_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    
    # Run comprehensive evaluation automatically after inference
    print(f"\n{'='*60}")
    print("STARTING COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*60}")
    
    model_id = "rag_finetuned" if args.use_finetuned else "rag_base"
    evaluation_metrics = evaluate_rag_model_performance(
        results, model_id=model_id, use_finetuned=args.use_finetuned
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
