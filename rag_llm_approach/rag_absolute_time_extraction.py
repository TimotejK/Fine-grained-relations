import os
# Disable torch compilation before importing torch to avoid dynamo issues
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import json
import pandas as pd
import torch
from rag_llm_approach.rag_config import RAGConfig, FULL_RAG

# Disable torch._dynamo to avoid issues with dynamic imports and debugging
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = False
    # Completely disable dynamo
    torch._dynamo.reset()
except Exception as e:
    print(f"Note: Could not configure torch._dynamo: {e}")

import argparse
from datetime import datetime
from difflib import get_close_matches
from unsloth import FastLanguageModel
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import compute_metrics, store_prediction_for_error_analysis
import hashlib

class RAGTemporalExtractor:
    def __init__(self, model_path="unsloth/gemma-3-12b-it-unsloth-bnb-4bit", use_finetuned=False, finetuned_model_path="/home/timotej/Documents/ai-programi/models/Gemma3_12B_FGTRE", max_seq_length=2048):
        self.model_path = model_path
        self.use_finetuned = use_finetuned
        self.finetuned_model_path = finetuned_model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the Gemma 3 12B model using unsloth for efficient inference"""
        try:
            if self.use_finetuned:
                # Load base model first, then load LoRA adapters
                print(f"Loading base model: {self.model_path}")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=self.max_seq_length,
                    dtype=None,  # Let unsloth choose the best dtype
                    load_in_4bit=True,
                )

                # Load LoRA adapters from the finetuned model directory
                print(f"Loading LoRA adapters from: {self.finetuned_model_path}")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.finetuned_model_path)
                print("LoRA adapters loaded successfully")
            else:
                # Load base model
                print(f"Loading base model: {self.model_path}")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=self.max_seq_length,
                    dtype=None,  # Let unsloth choose the best dtype
                    load_in_4bit=True,
                )

            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            print(f"Model loaded successfully (max_seq_length={self.max_seq_length})")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _tokenize_text(self, text, add_special_tokens=False):
        """
        Tokenize text handling both Gemma3Processor and standard tokenizers.
        Returns a list of token IDs.
        """
        try:
            # Try to access nested tokenizer (for processors)
            if hasattr(self.tokenizer, 'tokenizer'):
                return self.tokenizer.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            else:
                # Use processor/tokenizer directly with return_tensors='pt'
                tokenized = self.tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")
                return tokenized['input_ids'][0].tolist()
        except Exception as e:
            print(f"Warning: Tokenization error: {e}. Estimating token count.")
            # Fallback: rough estimation (4 chars per token)
            return list(range(len(text) // 4))

    def _reduce_rag_context(self, rag_context, max_timelines):
        """
        Reduce RAG context to keep only the first N timelines.
        Always preserves DURATION REFERENCE section.
        Returns reduced RAG context string.
        """
        if not rag_context or max_timelines <= 0:
            return ""

        lines = rag_context.split('\n')
        reduced_lines = []
        timeline_count = 0
        in_timeline = False
        in_duration_reference = False

        for line in lines:
            # Check if this is DURATION REFERENCE section (always keep)
            if line.strip().startswith("DURATION REFERENCE"):
                in_duration_reference = True
                reduced_lines.append(line)
            elif in_duration_reference:
                # Keep all lines in DURATION REFERENCE section
                reduced_lines.append(line)
            # Check if this is a timeline header
            elif line.strip().startswith("Timeline ") and "from document" in line:
                timeline_count += 1
                if timeline_count <= max_timelines:
                    in_timeline = True
                    reduced_lines.append(line)
                else:
                    in_timeline = False
            elif in_timeline or timeline_count == 0:
                # Keep line if we're in an included timeline or before any timeline
                reduced_lines.append(line)

        return '\n'.join(reduced_lines)

    def _truncate_prompt_intelligently(self, prompt):
        """
        Intelligently truncate the prompt if it's too long, preserving important parts.
        Uses a progressive truncation approach in a while loop.

        Truncation strategy (in order):
        1. Try full prompt (all RAG timelines)
        2. Reduce RAG to 2 timelines
        3. Reduce RAG to 1 timeline
        4. Remove RAG entirely
        5. Progressively truncate medical document

        Important parts (in order of preservation priority):
        1. Instructions and task description (always preserved)
        2. Event mention with surrounding context (always preserved)
        3. RAG context (reduced progressively)
        4. Start and end of the medical document (truncated last)

        Returns: truncated prompt that fits within max_seq_length
        """
        # Reserve tokens for generation (128) and some buffer for chat template
        max_prompt_tokens = self.max_seq_length - 128 - 30

        # First, try the full prompt without truncation
        current_prompt = prompt
        truncation_level = 0
        rag_reduction_applied = None  # Track RAG reduction: None, 2, 1, 0

        while True:
            # Format and tokenize to check length
            messages = [{"role": "user", "content": current_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Tokenize to check length
            tokens = self._tokenize_text(formatted_prompt, add_special_tokens=False)

            # Check if it fits
            if len(tokens) <= max_prompt_tokens:
                if truncation_level > 0 or rag_reduction_applied is not None:
                    msg = f"Prompt fits after "
                    if rag_reduction_applied is not None:
                        if rag_reduction_applied == 0:
                            msg += "removing RAG context"
                        else:
                            msg += f"reducing RAG to {rag_reduction_applied} timeline(s)"
                    if truncation_level > 0:
                        msg += f"{' and' if rag_reduction_applied is not None else ''} truncation level {truncation_level}"
                    msg += f" ({len(tokens)} tokens)"
                    print(msg)
                return current_prompt

            # Need to truncate
            if truncation_level == 0 and rag_reduction_applied is None:
                print(f"Prompt too long ({len(tokens)} > {max_prompt_tokens}). Starting RAG reduction...")
            else:
                print(f"Still too long ({len(tokens)} tokens). Trying next strategy...")

            # Parse the prompt to identify important sections
            context_start_marker = "Context (target event marked with <TARGET> tags):"
            task_marker = "\nTask: Predict the absolute start and end times"

            # RAG context markers - these appear AFTER the medical document, BEFORE task
            rag_markers = [
                "Relevant patient timelines with similar events:",
                "Relevant knowledge base entries about similar events:",
                "No similar events found in patient timelines"
            ]

            # Split prompt into sections
            parts = prompt.split(context_start_marker)
            if len(parts) != 2:
                # Fallback: simple progressive truncation
                print("Warning: Could not parse prompt structure, using simple progressive truncation")
                current_prompt = self._simple_progressive_truncate(prompt, max_prompt_tokens, truncation_level)
                if truncation_level > 10:
                    print("Warning: Reached maximum truncation level, returning best effort")
                    return current_prompt
                continue

            header = parts[0] + context_start_marker  # Instructions only
            remainder = parts[1]  # Contains: Medical doc + RAG context + Task

            # Split remainder to separate from task
            task_parts = remainder.split(task_marker)
            if len(task_parts) != 2:
                print("Warning: Could not find task section, using simple progressive truncation")
                current_prompt = self._simple_progressive_truncate(prompt, max_prompt_tokens, truncation_level)
                if truncation_level > 10:
                    print("Warning: Reached maximum truncation level, returning best effort")
                    return current_prompt
                continue

            # The part before task contains: Medical doc + RAG context
            before_task = task_parts[0].strip()
            task_section = task_marker + task_parts[1]

            # Extract RAG context which appears AFTER the medical document
            rag_context_original = ""
            medical_context = before_task

            # Check if RAG context exists (appears after medical doc, before task)
            for rag_marker in rag_markers:
                if rag_marker in before_task:
                    # Split into medical context and RAG context
                    split_parts = before_task.split(rag_marker, 1)
                    medical_context = split_parts[0].strip()
                    rag_context_original = "\n" + rag_marker + "\n" + split_parts[1].strip()
                    break

            # Strategy: Try RAG reduction before document truncation
            # Levels: None (full RAG) -> 2 timelines -> 1 timeline -> 0 timelines -> doc truncation
            rag_context = rag_context_original

            if rag_context_original and truncation_level == 0:
                # First attempt with full RAG - just iterate to try reduction
                truncation_level += 1
                continue
            elif rag_context_original and truncation_level == 1:
                # Try reducing to 2 timelines
                print("Reducing RAG context to 2 timelines...")
                rag_context = self._reduce_rag_context(rag_context_original, 2)
                rag_reduction_applied = 2
                truncation_level += 1
                current_prompt = header + " " + medical_context + rag_context + "\n\n" + task_section
                continue
            elif rag_context_original and truncation_level == 2:
                # Try reducing to 1 timeline
                print("Reducing RAG context to 1 timeline...")
                rag_context = self._reduce_rag_context(rag_context_original, 1)
                rag_reduction_applied = 1
                truncation_level += 1
                current_prompt = header + " " + medical_context + rag_context + "\n\n" + task_section
                continue
            elif rag_context_original and truncation_level == 3:
                # Try removing RAG entirely (full medical document, no RAG)
                print("Removing RAG context entirely, keeping full medical document...")
                rag_context = ""
                rag_reduction_applied = 0
                current_prompt = header + " " + medical_context + rag_context + "\n\n" + task_section
                truncation_level += 1
                continue

            # If we get here at level 4 or higher, RAG removal didn't help enough
            # Now start document truncation (only if level >= 4)
            if truncation_level < 4:
                # This shouldn't happen, but safeguard
                truncation_level = 4

            # Print message when starting document truncation
            if truncation_level == 4:
                print("Full document with no RAG still too long. Starting document truncation...")

            doc_truncation_level = truncation_level - 3  # Level 4 becomes doc truncation 1

            # Find the target event in the medical context (after removing RAG)
            target_start = medical_context.find("<TARGET>")
            target_end = medical_context.find("</TARGET>") + len("</TARGET>")

            if target_start == -1 or target_end == -1:
                print("Warning: Could not find <TARGET> markers, using simple progressive truncation")
                current_prompt = self._simple_progressive_truncate(prompt, max_prompt_tokens, truncation_level)
                if truncation_level > 10:
                    print("Warning: Reached maximum truncation level, returning best effort")
                    return current_prompt
                continue

            # Calculate how much space we have
            # RAG context is preserved, so count it with fixed tokens
            header_tokens = self._tokenize_text(header, add_special_tokens=False)
            task_tokens = self._tokenize_text(task_section, add_special_tokens=False)
            rag_tokens = self._tokenize_text(rag_context, add_special_tokens=False) if rag_context else []
            fixed_tokens = len(header_tokens) + len(task_tokens) + len(rag_tokens)

            available_for_context = max_prompt_tokens - fixed_tokens

            if available_for_context < 100:
                print(f"Warning: Very little space for medical context ({available_for_context} tokens available)")
                # Even with minimal space, we continue
                available_for_context = max(100, available_for_context)

            # Preserve important parts with progressive truncation of medical document
            # Use doc_truncation_level (which starts at 1 when we reach level 4)
            context_before_target = medical_context[:target_start]
            target_text = medical_context[target_start:target_end]
            context_after_target = medical_context[target_end:]

            # Calculate allocation based on doc_truncation_level
            # Start: 20% -> 15% -> 10% -> 5% -> 0%
            # Target: 60% -> 65% -> 70% -> 75% -> 80%
            # End: 20% -> 15% -> 10% -> 5% -> 0%
            chars_per_token = 4  # Rough estimate
            available_chars = max(available_for_context * chars_per_token, 400)  # Minimum 400 chars

            start_percent = max(0.20 - (truncation_level * 0.05), 0)
            end_percent = max(0.20 - (truncation_level * 0.05), 0)
            target_percent = min(1.0 - 0.40 + (truncation_level * 0.05), 1.0)

            start_chars = int(available_chars * start_percent)
            target_chars = int(available_chars * target_percent)
            end_chars = int(available_chars * end_percent)

            # Extract sections
            doc_start = context_before_target[:start_chars] if start_chars > 0 else ""
            doc_end = context_after_target[-end_chars:] if end_chars > 0 else ""

            # For target context, preserve surrounding text
            target_context_chars = target_chars // 2
            target_before = context_before_target[-target_context_chars:] if len(context_before_target) > target_context_chars else context_before_target
            target_after = context_after_target[:target_context_chars] if len(context_after_target) > target_context_chars else context_after_target

            # Reconstruct truncated medical context
            truncated_context = ""

            # Add start of document if we're keeping it
            if doc_start and doc_start != target_before:
                truncated_context += doc_start
                if len(context_before_target) > start_chars:
                    truncated_context += "\n[...truncated...]\n"

            # Add context around target (always preserved)
            truncated_context += target_before + target_text + target_after

            # Add end of document if we're keeping it
            if doc_end and doc_end != target_after:
                if len(context_after_target) > end_chars:
                    truncated_context += "\n[...truncated...]\n"
                truncated_context += doc_end

            # Reconstruct full prompt with RAG context after medical document, before task
            # Structure: header + medical_context + rag_context + task_section
            current_prompt = header + " " + truncated_context + rag_context + "\n\n" + task_section

            # Increment for next iteration
            truncation_level += 1

            # Safety check: if we've tried too many times, give up
            if truncation_level > 15:
                print("Warning: Reached maximum truncation level (15), returning best effort")
                return current_prompt


    def _simple_progressive_truncate(self, prompt, max_tokens, truncation_level):
        """
        Fallback: simple progressive truncation from the middle of the prompt.
        Each level removes more content from the middle.
        """
        chars_per_token = 4  # Rough estimate
        max_chars = max_tokens * chars_per_token

        if len(prompt) <= max_chars:
            return prompt

        # Progressively reduce the middle section
        # Level 1: Keep 40% start, 40% end
        # Level 2: Keep 35% start, 35% end
        # Level 3: Keep 30% start, 30% end
        # etc.
        keep_percent = max(0.40 - (truncation_level - 1) * 0.05, 0.10)

        keep_start = int(max_chars * keep_percent)
        keep_end = int(max_chars * keep_percent)

        truncated = prompt[:keep_start] + "\n[...content truncated...]\n" + prompt[-keep_end:]
        return truncated

    def call_llm(self, prompt):
        """Call the local Gemma 3 12B model for inference"""
        try:
            # Apply intelligent truncation if needed
            truncated_prompt = self._truncate_prompt_intelligently(prompt)

            # Format prompt for Gemma chat template
            messages = [{"role": "user", "content": truncated_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate (no truncation here since we already handled it)
            inputs = self.tokenizer(
                text=formatted_prompt,
                return_tensors="pt",
                truncation=False,  # We already truncated intelligently
                max_length=None
            ).to(self.model.device)

            # Verify input length
            input_length = inputs['input_ids'].shape[1]
            if input_length > self.max_seq_length:
                print(f"Warning: Input still too long ({input_length} > {self.max_seq_length}), applying hard truncation")
                inputs['input_ids'] = inputs['input_ids'][:, :self.max_seq_length]
                inputs['attention_mask'] = inputs['attention_mask'][:, :self.max_seq_length]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,     # Reduced from 0.7 for determinism
                    do_sample=False,     # Greedy decoding
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
            print(f"Parse error: {e}")
            return {"start": None, "end": None, "parse_error": str(e)}

class OpenAITemporalExtractor:
    """Temporal extractor using OpenAI API (GPT-4 or GPT-4.1)"""

    def __init__(self, model="gpt-5.2-2025-12-11", cache_dir="cache_openai"):
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Import OpenAI
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            print(f"OpenAI API initialized with model: {model}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

    def _get_cache_file_path(self, prompt):
        """Create a unique hash for the prompt to use as a filename"""
        prompt_hash = hashlib.sha256((prompt + self.model).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prompt_hash}.json")

    def _load_from_cache(self, cache_path):
        """Load response from cache if it exists"""
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_path, response):
        """Save response to cache"""
        with open(cache_path, "w") as f:
            json.dump(response, f, indent=2)

    def call_llm(self, prompt):
        """Call OpenAI API for inference"""
        try:
            # Check cache first
            cache_path = self._get_cache_file_path(prompt)
            cached_response = self._load_from_cache(cache_path)
            if cached_response:
                return cached_response

            # Make API call
            messages = [
                {"role": "system", "content": "You are a clinical temporal reasoning expert. Provide responses in valid JSON format."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

            response_text = response.choices[0].message.content

            # Parse the response
            parsed = self.parse_temporal_response(response_text)

            # Cache the result
            self._save_to_cache(cache_path, parsed)

            return parsed

        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
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

class GeminiTemporalExtractor:
    """Temporal extractor using Google Gemini API with rate limiting"""

    def __init__(self, model="gemini-3-flash-preview", cache_dir="cache_gemini", requests_per_minute=15):
        self.model = model
        self.cache_dir = cache_dir
        self.requests_per_minute = requests_per_minute
        os.makedirs(self.cache_dir, exist_ok=True)

        # Rate limiting setup
        import time
        from collections import deque
        self.request_times = deque()
        self.time = time

        # Import Gemini
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self.client = genai.Client(api_key=api_key)
            print(f"Gemini API initialized with model: {model}")
            print(f"Rate limiting: {requests_per_minute} requests per minute")
        except ImportError:
            raise ImportError("Google GenAI library not installed. Install with: pip install google-genai")

    def _get_cache_file_path(self, prompt):
        """Create a unique hash for the prompt to use as a filename"""
        prompt_hash = hashlib.sha256((prompt + self.model).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prompt_hash}.json")

    def _load_from_cache(self, cache_path):
        """Load response from cache if it exists"""
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_path, response):
        """Save response to cache"""
        with open(cache_path, "w") as f:
            json.dump(response, f, indent=2)

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limiting"""
        current_time = self.time.time()

        # Remove timestamps older than 60 seconds
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        # If we've made too many requests in the last minute, wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0]) + 0.1  # Add small buffer
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                self.time.sleep(sleep_time)
                # Clear old timestamps after waiting
                current_time = self.time.time()
                while self.request_times and current_time - self.request_times[0] > 60:
                    self.request_times.popleft()

        # Record this request
        self.request_times.append(current_time)

    def call_llm(self, prompt):
        """Call Gemini API for inference"""
        try:
            # Check cache first
            cache_path = self._get_cache_file_path(prompt)
            cached_response = self._load_from_cache(cache_path)
            if cached_response:
                return cached_response

            # Wait for rate limiting before making API call
            self._wait_for_rate_limit()

            # Make API call
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                    "max_output_tokens": 2048
                }
            )

            # Check if response was truncated
            if response.candidates and response.candidates[0].finish_reason.name == 'MAX_TOKENS':
                print(f"Warning: Response truncated due to MAX_TOKENS limit")
                return {"start": None, "end": None, "error": "Response truncated - MAX_TOKENS reached"}

            response_text = response.text

            # Check if response text is empty or None
            if not response_text:
                error_msg = "Empty response from Gemini API"
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason.name
                    error_msg += f" (finish_reason: {finish_reason})"
                print(f"Error: {error_msg}")
                return {"start": None, "end": None, "error": error_msg}

            # Parse the response
            parsed = self.parse_temporal_response(response_text)

            # Cache the result
            self._save_to_cache(cache_path, parsed)

            return parsed

        except Exception as e:
            print(f"Error in Gemini API call: {e}")
            # Add more detailed error information
            import traceback
            traceback.print_exc()
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

# Load patient timelines
def load_patient_timelines(timeline_path):
    with open(timeline_path, 'r') as f:
        data = json.load(f)
    return data['timelines']

# Retrieve relevant KB entries for an event
def retrieve_relevant_kb_entries(event_text, kb_events, top_k=3):
    kb_texts = [e['event_text'] for e in kb_events]
    matches = get_close_matches(event_text, kb_texts, n=top_k, cutoff=0.5)
    return [e for e in kb_events if e['event_text'] in matches]

# Retrieve relevant timeline events for an event
def retrieve_relevant_timeline_events(event_text, timelines, top_k=3):
    """
    Find timelines containing similar events to the query event.
    Returns surrounding events from those timelines for context.
    """
    relevant_timeline_events = []

    for timeline in timelines:
        events = timeline.get('events', [])
        event_texts = [e['event_text'] for e in events]

        # Find matching events in this timeline
        matches = get_close_matches(event_text, event_texts, n=1, cutoff=0.5)

        if matches:
            # Found a similar event in this timeline
            matching_event_idx = event_texts.index(matches[0])
            matching_event = events[matching_event_idx]

            # Get surrounding events for context (before and after)
            start_idx = max(0, matching_event_idx - 2)
            end_idx = min(len(events), matching_event_idx + 3)

            surrounding_events = {
                'matching_event': matching_event,
                'timeline_context': events[start_idx:end_idx],
                'document_id': timeline.get('document_id', 'unknown'),
                'sequence_position': matching_event_idx
            }

            relevant_timeline_events.append(surrounding_events)

            if len(relevant_timeline_events) >= top_k:
                break

    return relevant_timeline_events

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

# Construct a prompt without RAG context (no KB or timeline retrieval)
def construct_no_rag_prompt(event_row):
    """
    Construct a prompt without any RAG context - just the event and its context.
    This tests the model's base temporal reasoning capabilities.
    """
    event_text = event_row.get('event_text', event_row.get('event', ''))
    context = event_row.get('context', '')

    # Mark the target event in the context with XML tags
    if event_text and event_text in context:
        marked_context = context.replace(event_text, f"<TARGET>{event_text}</TARGET>", 1)
    else:
        marked_context = context

    prompt = (
        f"You are a clinical temporal reasoning expert. Your task is to predict the start and end times of medical events.\n\n"
        f"Event to analyze: {event_text}\n"
        f"Context (target event marked with <TARGET> tags): {marked_context}\n\n"
        f"Task: Predict the absolute start and end times for the target event marked with <TARGET> tags. You MUST provide specific temporal predictions.\n\n"
        "Guidelines:\n"
        "1. If the time is explicitly mentioned in the context, use that information.\n"
        "2. If the time is NOT explicitly mentioned, infer the most likely time based on:\n"
        "   - The event type and its typical timing patterns\n"
        "   - Clinical context and typical medical practice patterns\n"
        "   - Temporal relations to other events mentioned in the context\n"
        "3. Consider typical durations for different event types when predicting start and end times.\n"
        "4. Focus on the event marked with <TARGET> tags in the context.\n\n"
        "Return your answer as a JSON object with 'start' and 'end' fields in ISO format (YYYY-MM-DDTHH:MM:SS).\n"
        "Example: {\"start\": \"1900-01-15T09:30:00\", \"end\": \"1900-01-15T11:00:00\"}\n\n"
        "You must provide concrete temporal predictions even if the exact time is not stated in the text."
    )

    return prompt

# Construct a RAG prompt using patient timelines
def construct_timeline_rag_prompt(event_row, timeline_events):
    """
    Construct a RAG prompt using extracted patient timelines.
    Provides surrounding events from similar timelines for better temporal context.
    The target event in the discharge summary context is marked with <TARGET> XML tags.
    Explicitly includes durations of similar events as a reference.
    """
    event_text = event_row.get('event_text', event_row.get('event', ''))
    context = event_row.get('context', '')

    # Mark the target event in the context with XML tags
    if event_text and event_text in context:
        marked_context = context.replace(event_text, f"<TARGET>{event_text}</TARGET>", 1)
    else:
        marked_context = context

    prompt = (
        f"You are a clinical temporal reasoning expert. Your task is to predict the start and end times of medical events.\n\n"
        f"Event to analyze: {event_text}\n"
        f"Context (target event marked with <TARGET> tags): {marked_context}\n\n"
    )

    # Collect durations from similar events for explicit reference
    similar_durations = []

    if timeline_events:
        prompt += "Relevant patient timelines with similar events:\n\n"

        for idx, timeline_info in enumerate(timeline_events, 1):
            matching_event = timeline_info['matching_event']
            context_events = timeline_info['timeline_context']

            prompt += f"Timeline {idx} (from document {timeline_info['document_id']}):\n"
            prompt += f"  Similar event found: '{matching_event['event_text']}' (Type: {matching_event['event_type']})\n"

            if matching_event.get('absolute_time'):
                prompt += f"    Absolute time: {matching_event['absolute_time']}\n"
            if matching_event.get('temporal_expression'):
                prompt += f"    Temporal expression: {matching_event['temporal_expression']}\n"
            if matching_event.get('duration'):
                duration_str = matching_event['duration']
                prompt += f"    Duration: {duration_str}\n"
                # Collect for reference section
                similar_durations.append({
                    'event': matching_event['event_text'],
                    'duration': duration_str,
                    'duration_minutes': matching_event.get('duration_minutes')
                })

            prompt += f"\n  Surrounding events in this timeline:\n"
            for evt in context_events:
                evt_info = f"    - {evt['event_text']} (Type: {evt['event_type']})"
                if evt.get('absolute_time'):
                    evt_info += f" | Time: {evt['absolute_time']}"
                elif evt.get('temporal_expression'):
                    evt_info += f" | Temporal expr: {evt['temporal_expression']}"
                if evt.get('duration'):
                    evt_info += f" | Duration: {evt['duration']}"
                if evt.get('sequence_position') is not None:
                    evt_info += f" | Position: {evt['sequence_position']}"
                prompt += evt_info + "\n"
            prompt += "\n"

        # Add explicit duration reference section if we found similar events with durations
        if similar_durations:
            prompt += "DURATION REFERENCE - Similar events had the following durations:\n"
            for dur_info in similar_durations:
                prompt += f"  - '{dur_info['event']}': {dur_info['duration']}"
                if dur_info.get('duration_minutes'):
                    prompt += f" ({dur_info['duration_minutes']} minutes)"
                prompt += "\n"
            prompt += "\nUse these durations as a reference as the difference between the end and start time should represent the event duration.\n\n"
    else:
        prompt += "No similar events found in patient timelines. Use general medical knowledge.\n\n"

    prompt += (
        "\nTask: Predict the absolute start and end times for the target event marked with <TARGET> tags. You MUST provide specific temporal predictions.\n\n"
        "Guidelines:\n"
        "1. If the time is explicitly mentioned in the context, use that information.\n"
        "2. If the time is NOT explicitly mentioned, infer the most likely time based on:\n"
        "   - The event type and its typical timing patterns\n"
        "   - Similar events from the patient timelines above\n"
        "   - The sequence and temporal relationships of surrounding events\n"
        "   - Clinical context and typical medical practice patterns\n"
        "3. **IMPORTANT: Consider the durations of similar events provided in the DURATION REFERENCE section above.**\n"
        "   The target event should have a similar duration unless there are specific reasons to differ.\n"
        "4. Use the temporal patterns from similar timelines to guide your predictions.\n"
        "5. Focus on the event marked with <TARGET> tags in the context.\n"
        "6. Ensure the predicted end time is consistent with the start time and expected duration.\n\n"
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

        # Return with consistent formatting (no extra spaces)
        return json.dumps(response, separators=(',', ':'))

    except Exception as e:
        # Consistent fallback format
        return '{"start":"1900-01-01T00:00:00","end":"1900-01-01T00:00:00"}'

def generate_finetuning_dataset(use_timelines=False, no_rag=False, timeline_file='rag_llm_approach/patient_timelines.json'):
    """Generate QA pairs for fine-tuning the RAG model"""
    print(f"Generating fine-tuning dataset (use_timelines={use_timelines}, no_rag={no_rag})...")
    if use_timelines:
        print(f"Using timeline file: {timeline_file}")
    if no_rag:
        print("Running in NO RAG mode - no context retrieval will be used")

    # Load training data
    df = load_i2b2_absolute_data(test_split=False)

    # Initialize variables
    kb_events = []
    timelines = []

    # Load event knowledge base or patient timelines (unless no_rag is True)
    if no_rag:
        print("Skipping RAG context loading (no_rag mode)")
    elif use_timelines:
        try:
            timelines = load_patient_timelines(timeline_file)
            print(f"Loaded {len(timelines)} patient timelines")
        except FileNotFoundError:
            print(f"Warning: Patient timelines not found at {timeline_file}, falling back to KB mode")
            use_timelines = False
            try:
                kb_events = load_event_kb('rag_llm_approach/temporal_events_kb.json')
            except FileNotFoundError:
                print("Warning: Knowledge base not found, using empty KB")
                kb_events = []
    else:
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

            # Generate prompt based on mode
            if no_rag:
                prompt = construct_no_rag_prompt(
                    {'event_text': event_string, 'context': row['text']}
                )
            elif use_timelines:
                timeline_events = retrieve_relevant_timeline_events(event_string, timelines)
                prompt = construct_timeline_rag_prompt(
                    {'event_text': event_string, 'context': row['text']},
                    timeline_events
                )
            else:
                kb_entries = retrieve_relevant_kb_entries(event_string, kb_events)
                prompt = construct_rag_prompt(
                    {'event_text': event_string, 'context': row['text']},
                    kb_entries
                )

            # Generate ground truth response
            ground_truth = generate_ground_truth_response(row)

            # Determine instruction text based on mode
            if no_rag:
                instruction = 'Extract temporal information for clinical events using only the provided context.'
            elif use_timelines:
                instruction = 'Extract temporal information for clinical events using patient timelines.'
            else:
                instruction = 'Extract temporal information for clinical events using provided knowledge base.'

            qa_pairs.append({
                'instruction': instruction,
                'input': prompt,
                'output': ground_truth,
                'metadata': {
                    'document_id': row['document_id'],
                    'event_text': event_string,
                    'start_char': row['start_char'],
                    'end_char': row['end_char'],
                    'mode': 'no_rag' if no_rag else ('timeline' if use_timelines else 'kb')
                }
            })

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} examples")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Save dataset
    if no_rag:
        suffix = '_no_rag'
    elif use_timelines:
        suffix = '_timeline'
    else:
        suffix = ''
    output_path = f'rag_llm_approach/rag_finetuning_dataset{suffix}.json'
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
        
        # Error analysis is automatically saved by store_prediction_for_error_analysis

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

def run_all_configurations(timeline_file='rag_llm_approach/i2b2_patient_timelines.json', max_examples=None, subsample_size=1000, random_seed=42):
    """
    Run inference and evaluation for all configurations including RAG ablation studies.

    Configurations include:
    1. No RAG + Base/Finetuned Model
    2. KB RAG + Base/Finetuned Model
    3. Timeline RAG + Base/Finetuned Model (Full information)
    4. Timeline RAG + Base/Finetuned Model (Duration only)
    5. Timeline RAG + Base/Finetuned Model (Duration + Time intervals)
    6. Timeline RAG + Base/Finetuned Model (No duration)
    7. Timeline RAG + OpenAI/Gemini (subsample)

    Args:
        timeline_file: Path to patient timelines JSON file
        max_examples: Maximum number of examples to process (for testing)
        subsample_size: If set, use random subsample of this size for API models
        random_seed: Random seed for reproducible subsampling
    """
    configurations = [
        # === BASE CONFIGURATIONS ===
        # {'name': 'No RAG + Base Gemma 3 12B', 'use_finetuned': False, 'use_timelines': False, 'no_rag': True, 'model_type': 'local'},
        {'name': 'Timeline RAG (Full) + Base Gemma 3 12B', 'use_finetuned': False, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'full'},
        {'name': 'Timeline RAG (Full) + Finetuned Gemma 3 12B', 'use_finetuned': True, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'full'},

        # === ABLATION STUDY: DURATION INFORMATION ===
        {'name': 'Timeline RAG (Duration Only) + Base Gemma 3 12B', 'use_finetuned': False, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'duration_only'},
        {'name': 'Timeline RAG (Duration + Intervals) + Base Gemma 3 12B', 'use_finetuned': False, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'duration_with_intervals'},
        {'name': 'Timeline RAG (No Duration) + Base Gemma 3 12B', 'use_finetuned': False, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'no_duration'},

        # === FINETUNED ABLATION ===
        {'name': 'Timeline RAG (Duration Only) + Finetuned Gemma 3 12B', 'use_finetuned': True, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'duration_only'},
        {'name': 'Timeline RAG (Duration + Intervals) + Finetuned Gemma 3 12B', 'use_finetuned': True, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'duration_with_intervals'},
        {'name': 'Timeline RAG (No Duration) + Finetuned Gemma 3 12B', 'use_finetuned': True, 'use_timelines': True, 'no_rag': False, 'model_type': 'local', 'rag_config': 'no_duration'},

        # === API MODELS (commented out to avoid costs) ===
        # {'name': 'Timeline RAG + OpenAI GPT-5.2', 'use_finetuned': False, 'use_timelines': True, 'no_rag': False, 'model_type': 'openai', 'use_subsample': True, 'rag_config': 'full'},
        # {'name': 'Timeline RAG + Gemini', 'use_finetuned': False, 'use_timelines': True, 'no_rag': False, 'model_type': 'gemini', 'use_subsample': True, 'rag_config': 'full'},
    ]

    all_results = {}

    print("="*80)
    print("RUNNING COMPREHENSIVE EVALUATION WITH RAG ABLATION STUDY")
    print("="*80)
    print(f"Total configurations to test: {len(configurations)}")
    print("\nRAG Configuration Options:")
    print("  - full: All information (baseline)")
    print("  - duration_only: Only duration information from similar events")
    print("  - duration_with_intervals: Durations + time intervals between events")
    print("  - no_duration: All information except durations")
    if subsample_size:
        print(f"\nAPI models will use random subsample of {subsample_size} examples (seed={random_seed})")
    print("="*80)

    for config_idx, config in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_idx}/{len(configurations)}: {config['name']}")
        print(f"{'='*80}")
        rag_config_name = config.get('rag_config', 'full')
        print(f"Settings: model_type={config.get('model_type', 'local')}, use_timelines={config['use_timelines']}, no_rag={config['no_rag']}, rag_config={rag_config_name}")
        print(f"{'='*80}\n")

        try:
            # Initialize extractor based on model type
            global rag_extractor
            model_type = config.get('model_type', 'local')

            if model_type == 'openai':
                rag_extractor = OpenAITemporalExtractor()
            elif model_type == 'gemini':
                rag_extractor = GeminiTemporalExtractor()
            else:
                rag_extractor = RAGTemporalExtractor(use_finetuned=config['use_finetuned'])

            # Load test data
            df = load_i2b2_absolute_data(test_split=True)

            # Apply subsampling for API models to reduce costs
            if config.get('use_subsample', False) and subsample_size:
                print(f"Applying random subsample of {subsample_size} examples for API model...")
                df = df.sample(n=min(subsample_size, len(df)), random_state=random_seed)
                print(f"Subsampled dataset size: {len(df)}")
            elif max_examples:
                df = df[:max_examples]
                print(f"Limited to {max_examples} examples for testing")

            # Initialize variables
            kb_events = []
            timelines = []

            # Load event knowledge base or patient timelines based on mode
            if config['no_rag']:
                print("Running in NO RAG mode - no context retrieval will be used")
            elif config['use_timelines']:
                try:
                    timelines = load_patient_timelines(timeline_file)
                    print(f"Loaded {len(timelines)} patient timelines for RAG")
                except FileNotFoundError:
                    print(f"Warning: Patient timelines not found at {timeline_file}, skipping this configuration")
                    continue
            else:
                try:
                    kb_events = load_event_kb('rag_llm_approach/temporal_events_kb.json')
                    print(f"Loaded knowledge base with {len(kb_events)} events")
                except FileNotFoundError:
                    print("Warning: Knowledge base not found, using empty KB")
                    kb_events = []

            # Run inference
            results = []
            timeline_stats = []  # Track timeline retrieval statistics

            i = 0
            for idx, row in df.iterrows():
                i += 1
                try:
                    # Extract event string from text using start_char and end_char
                    event_string = row['text'][row['start_char']:row['end_char']]

                    # Generate prompt based on mode and track statistics
                    num_similar_events = 0
                    retrieval_info = {}

                    if config['no_rag']:
                        prompt = construct_no_rag_prompt(
                            {'event_text': event_string, 'context': row['text']}
                        )
                        num_similar_events = 0
                        retrieval_info = {'mode': 'no_rag'}
                    elif config['use_timelines']:
                        timeline_events = retrieve_relevant_timeline_events(event_string, timelines)
                        num_similar_events = len(timeline_events)

                        # Collect detailed retrieval info
                        retrieval_info = {
                            'mode': 'timeline',
                            'num_similar_events': num_similar_events,
                            'similar_events': [
                                {
                                    'document_id': te['document_id'],
                                    'matching_event': te['matching_event']['event_text'],
                                    'num_context_events': len(te['timeline_context'])
                                }
                                for te in timeline_events
                            ]
                        }

                        prompt = construct_timeline_rag_prompt(
                            {'event_text': event_string, 'context': row['text']},
                            timeline_events
                        )
                    else:
                        kb_entries = retrieve_relevant_kb_entries(event_string, kb_events)
                        num_similar_events = len(kb_entries)
                        retrieval_info = {
                            'mode': 'kb',
                            'num_similar_events': num_similar_events
                        }
                        prompt = construct_rag_prompt(
                            {'event_text': event_string, 'context': row['text']},
                            kb_entries
                        )

                    llm_response = rag_extractor.call_llm(prompt)

                    results.append({
                        'document_id': row['document_id'],
                        'event_text': event_string,
                        'llm_response': llm_response,
                        'ground_truth': {
                            'start_time_minutes': row['start_time_minutes'],
                            'start_upper_minutes': row['start_upper_minutes'],
                            'start_lower_minutes': row['start_lower_minutes'],
                            'end_time_minutes': row['end_time_minutes'],
                            'end_upper_minutes': row['end_upper_minutes'],
                            'end_lower_minutes': row['end_lower_minutes'],
                            'duration_minutes': row['duration_minutes'],
                            'duration_lower_minutes': row['duration_lower_minutes'],
                            'duration_upper_minutes': row['duration_upper_minutes']
                        }
                    })

                    # Track timeline statistics
                    timeline_stats.append({
                        'document_id': row['document_id'],
                        'event_text': event_string,
                        'num_similar_events_found': num_similar_events,
                        'retrieval_info': retrieval_info
                    })

                    if idx % 10 == 0:
                        print(f"Processed {i}/{len(df)} test examples")

                except Exception as e:
                    print(f"Error processing test row {i}: {e}")
                    continue

            # Save results
            model_type = config.get('model_type', 'local')
            if config['no_rag']:
                mode_suffix = '_no_rag'
            elif config['use_timelines']:
                mode_suffix = '_timeline'
            else:
                mode_suffix = ''

            if model_type == 'openai':
                model_suffix = '_openai'
            elif model_type == 'gemini':
                model_suffix = '_gemini'
            else:
                finetuned_suffix = '_finetuned' if config['use_finetuned'] else ''
                model_suffix = finetuned_suffix

            output_file = f'rag_llm_approach/results/rag_absolute_time_predictions{model_suffix}{mode_suffix}.json'
            stats_file = f'rag_llm_approach/results/retrieval_stats{model_suffix}{mode_suffix}.json'

            # Create results directory if it doesn't exist
            os.makedirs('rag_llm_approach/results', exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Save timeline retrieval statistics
            if timeline_stats:
                # Compute summary statistics
                total_events = len(timeline_stats)
                events_with_similar = sum(1 for stat in timeline_stats if stat['num_similar_events_found'] > 0)
                avg_similar_events = sum(stat['num_similar_events_found'] for stat in timeline_stats) / total_events if total_events > 0 else 0

                stats_summary = {
                    'configuration': config['name'],
                    'total_events_processed': total_events,
                    'events_with_similar_found': events_with_similar,
                    'events_without_similar': total_events - events_with_similar,
                    'percentage_with_similar': (events_with_similar / total_events * 100) if total_events > 0 else 0,
                    'average_similar_events_per_query': avg_similar_events,
                    'detailed_stats': timeline_stats
                }

                with open(stats_file, 'w') as f:
                    json.dump(stats_summary, f, indent=2)

                print(f"\n✓ Retrieval statistics saved to: {stats_file}")
                print(f"  Events with similar found: {events_with_similar}/{total_events} ({stats_summary['percentage_with_similar']:.1f}%)")
                print(f"  Average similar events per query: {avg_similar_events:.2f}")

            print(f"✓ Results saved to: {output_file}")
            print(f"  Total predictions: {len(results)}")

            # Store results for final summary (no evaluation, just file info)
            all_results[config['name']] = {
                'config': config,
                'output_file': output_file,
                'total_predictions': len(results)
            }

            print(f"\n{'='*60}")
            print(f"COMPLETED: {config['name']}")
            print(f"{'='*60}\n")

            # Clean up model to free memory before loading next one
            if rag_extractor:
                # Only delete model/tokenizer for local models
                if hasattr(rag_extractor, 'model'):
                    del rag_extractor.model
                if hasattr(rag_extractor, 'tokenizer'):
                    del rag_extractor.tokenizer
                del rag_extractor
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n!!! ERROR in configuration '{config['name']}': {e}")
            print(f"Skipping to next configuration...\n")
            import traceback
            traceback.print_exc()
            all_results[config['name']] = {
                'config': config,
                'error': str(e)
            }
            # Clean up on error too
            try:
                if 'rag_extractor' in locals() and rag_extractor:
                    if hasattr(rag_extractor, 'model'):
                        del rag_extractor.model
                    if hasattr(rag_extractor, 'tokenizer'):
                        del rag_extractor.tokenizer
                    del rag_extractor
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except:
                pass
            continue

    # Print final comparison
    print("\n" + "="*80)
    print("FINAL SUMMARY OF ALL CONFIGURATIONS")
    print("="*80)

    for config_name, result in all_results.items():
        print(f"\n{config_name}:")
        if 'error' in result:
            print(f"  ❌ ERROR: {result['error']}")
        else:
            print(f"  ✓ Predictions collected: {result['total_predictions']}")
            print(f"  ✓ Results file: {result['output_file']}")

    # Save comprehensive comparison
    comparison_file = f"rag_llm_approach/results/run_all_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('rag_llm_approach/results', exist_ok=True)
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Summary saved to: {comparison_file}")
    print(f"{'='*80}")
    print(f"\nAll prediction files have been saved to: rag_llm_approach/results/")
    print(f"Use the evaluate_all_results.py script to compute metrics from these files.")
    print(f"{'='*80}\n")

    return all_results

def main():
    parser = argparse.ArgumentParser(description='RAG Temporal Extraction')
    parser.add_argument('--mode', choices=['inference', 'generate_data', 'run_all'],
                       default='inference', help='Mode to run the system')
    parser.add_argument('--use_finetuned', action='store_true',
                       help='Use fine-tuned model for inference')
    parser.add_argument('--use_timelines', action='store_true',
                       help='Use patient timelines instead of knowledge base for RAG')
    parser.add_argument('--no_rag', action='store_true',
                       help='Do not use any RAG context - test base model capabilities')
    parser.add_argument('--timeline_file',
                       default='rag_llm_approach/patient_timelines.json',
                       help='Path to patient timelines JSON file (default: patient_timelines.json)')
    parser.add_argument('--evaluate', action='store_true',
                          help='Run comprehensive evaluation after inference')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Limit number of test examples (useful for testing)')
    parser.add_argument('--subsample_size', type=int, default=100,
                       help='Random subsample size for API models to reduce costs (default: 100)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible subsampling (default: 42)')

    args = parser.parse_args()

    if args.mode == 'generate_data':
        generate_finetuning_dataset(use_timelines=args.use_timelines, no_rag=args.no_rag, timeline_file=args.timeline_file)
        return

    if args.mode == 'run_all':
        run_all_configurations(
            timeline_file=args.timeline_file,
            max_examples=args.max_examples,
            subsample_size=args.subsample_size,
            random_seed=args.random_seed
        )
        return

    # Inference mode
    print(f"Running inference mode, use_finetuned: {args.use_finetuned}, use_timelines: {args.use_timelines}, no_rag: {args.no_rag}")
    if args.use_timelines:
        print(f"Timeline file: {args.timeline_file}")

    # Initialize RAG extractor
    global rag_extractor
    rag_extractor = RAGTemporalExtractor(use_finetuned=args.use_finetuned)

    # Load test data
    df = load_i2b2_absolute_data(test_split=True)
    df = df[:10]

    # Initialize variables
    kb_events = []
    timelines = []

    # Load event knowledge base or patient timelines based on mode (unless no_rag is True)
    if args.no_rag:
        print("Running in NO RAG mode - no context retrieval will be used")
    elif args.use_timelines:
        try:
            timelines = load_patient_timelines(args.timeline_file)
            print(f"Loaded {len(timelines)} patient timelines for RAG")
        except FileNotFoundError:
            print(f"Warning: Patient timelines not found at {args.timeline_file}, falling back to KB mode")
            args.use_timelines = False
            try:
                kb_events = load_event_kb('rag_llm_approach/temporal_events_kb.json')
                print(f"Loaded knowledge base with {len(kb_events)} events")
            except FileNotFoundError:
                print("Warning: Knowledge base not found, using empty KB")
                kb_events = []
    else:
        try:
            kb_events = load_event_kb('rag_llm_approach/temporal_events_kb.json')
            print(f"Loaded knowledge base with {len(kb_events)} events")
        except FileNotFoundError:
            print("Warning: Knowledge base not found, using empty KB")
            kb_events = []

    results = []
    timeline_stats = []  # Track timeline retrieval statistics

    for idx, row in df.iterrows():
        try:
            # Extract event string from text using start_char and end_char
            event_string = row['text'][row['start_char']:row['end_char']]

            # Generate prompt based on mode and track statistics
            num_similar_events = 0
            retrieval_info = {}

            if args.no_rag:
                prompt = construct_no_rag_prompt(
                    {'event_text': event_string, 'context': row['text']}
                )
                num_similar_events = 0
                retrieval_info = {'mode': 'no_rag'}
            elif args.use_timelines:
                timeline_events = retrieve_relevant_timeline_events(event_string, timelines)
                num_similar_events = len(timeline_events)

                # Collect detailed retrieval info
                retrieval_info = {
                    'mode': 'timeline',
                    'num_similar_events': num_similar_events,
                    'similar_events': [
                        {
                            'document_id': te['document_id'],
                            'matching_event': te['matching_event']['event_text'],
                            'num_context_events': len(te['timeline_context'])
                        }
                        for te in timeline_events
                    ]
                }

                prompt = construct_timeline_rag_prompt(
                    {'event_text': event_string, 'context': row['text']},
                    timeline_events
                )
            else:
                kb_entries = retrieve_relevant_kb_entries(event_string, kb_events)
                num_similar_events = len(kb_entries)
                retrieval_info = {
                    'mode': 'kb',
                    'num_similar_events': num_similar_events
                }
                prompt = construct_rag_prompt(
                    {'event_text': event_string, 'context': row['text']},
                    kb_entries
                )

            llm_response = rag_extractor.call_llm(prompt)

            results.append({
                'document_id': row['document_id'],
                'event_text': event_string,
                'llm_response': llm_response,
                'ground_truth': {
                    'start_time_minutes': row['start_time_minutes'],
                    'start_upper_minutes': row['start_upper_minutes'],
                    'start_lower_minutes': row['start_lower_minutes'],
                    'end_time_minutes': row['end_time_minutes'],
                    'end_upper_minutes': row['end_upper_minutes'],
                    'end_lower_minutes': row['end_lower_minutes'],
                    'duration_minutes': row['duration_minutes'],
                    'duration_lower_minutes': row['duration_lower_minutes'],
                    'duration_upper_minutes': row['duration_upper_minutes']
                }
            })

            # Track timeline statistics
            timeline_stats.append({
                'document_id': row['document_id'],
                'event_text': event_string,
                'num_similar_events_found': num_similar_events,
                'retrieval_info': retrieval_info
            })

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(df)} test examples")

        except Exception as e:
            print(f"Error processing test row {idx}: {e}")
            continue

    # Save results
    if args.no_rag:
        mode_suffix = '_no_rag'
    elif args.use_timelines:
        mode_suffix = '_timeline'
    else:
        mode_suffix = ''
    finetuned_suffix = '_finetuned' if args.use_finetuned else ''
    output_file = f'rag_llm_approach/rag_absolute_time_predictions{finetuned_suffix}{mode_suffix}.json'
    stats_file = f'rag_llm_approach/retrieval_stats{finetuned_suffix}{mode_suffix}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    
    # Save timeline retrieval statistics
    if timeline_stats:
        # Compute summary statistics
        total_events = len(timeline_stats)
        events_with_similar = sum(1 for stat in timeline_stats if stat['num_similar_events_found'] > 0)
        avg_similar_events = sum(stat['num_similar_events_found'] for stat in timeline_stats) / total_events if total_events > 0 else 0

        stats_summary = {
            'mode': 'timeline' if args.use_timelines else ('kb' if not args.no_rag else 'no_rag'),
            'use_finetuned': args.use_finetuned,
            'total_events_processed': total_events,
            'events_with_similar_found': events_with_similar,
            'events_without_similar': total_events - events_with_similar,
            'percentage_with_similar': (events_with_similar / total_events * 100) if total_events > 0 else 0,
            'average_similar_events_per_query': avg_similar_events,
            'detailed_stats': timeline_stats
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_summary, f, indent=2)

        print(f"Retrieval statistics saved to: {stats_file}")
        print(f"  Events with similar found: {events_with_similar}/{total_events} ({stats_summary['percentage_with_similar']:.1f}%)")
        print(f"  Average similar events per query: {avg_similar_events:.2f}")

    # Run comprehensive evaluation automatically after inference
    print(f"\n{'='*60}")
    print("STARTING COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*60}")
    
    if args.no_rag:
        mode_id = "no_rag"
    elif args.use_timelines:
        mode_id = "timeline"
    else:
        mode_id = "kb"
    model_id = f"rag_{mode_id}_{'finetuned' if args.use_finetuned else 'base'}"
    evaluation_metrics = evaluate_rag_model_performance(
        results, model_id=model_id, use_finetuned=args.use_finetuned
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")
    print(f"{'='*60}")

if __name__ == '__main__':
    run_all_configurations()
