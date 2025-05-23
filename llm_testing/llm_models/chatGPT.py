import os
import hashlib
import json
import openai
from openai import OpenAI
from pydantic.v1 import BaseModel

from llm_testing.absolute_time_predictor import TimeInterval


class OpenAIModel:
    def __init__(self, api_key, cache_dir="cache_openai", model="gpt-4-1106-preview"):
        self.client = OpenAI(api_key=api_key)
        self.cache_dir = cache_dir
        self.model = model
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_file_path(self, prompt):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prompt_hash}.json")

    def _load_from_cache(self, cache_path):
        # Load response from cache if it exists
        from llm_testing.absolute_time_predictor import TimeInterval

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                try:
                    return TimeInterval.model_validate(json.load(f))
                except ValueError:
                    return f.read()
        return None

    def _save_to_cache(self, cache_path, response):
        # Save response to cache
        with open(cache_path, "w") as f:
            if type(response) == str:
                f.write(response)
            else:
                f.write(response.model_dump_json())

    def _call_openai_chat_json(self, messages, structure: BaseModel):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=structure,  # structured output
        )
        return response.choices[0].message.content
    def _call_openai_chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def predict(self, prompt):
        cache_path = self._get_cache_file_path(prompt)
        cached_response = self._load_from_cache(cache_path)
        if cached_response:
            print("Loaded from cache.")
            return cached_response

        messages = [
            {"role": "system", "content": "You are a helpful assistant that predicts event times in the format 'PREDICTION: value unit direction' (e.g. 'PREDICTION: 3 minutes before')."},
            {"role": "user", "content": prompt},
        ]

        response = self._call_openai_chat(messages)
        self._save_to_cache(cache_path, response)
        return response

    def predict_schema(self, prompt, schema):
        response = self.client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "You are a medical language model specializing in clinical event extraction and temporal reasoning. " +
                 "Your task is to analyze patient discharge summaries containing annotated events (e.g., <event1>, <event2>, etc.). " +
                 "For each annotated event, infer when the event started and ended relative to the admission date. " +
                 "Output must be a strictly formatted JSON array, with time deltas expressed in years, months, days, hours, and minutes. " +
                 "If information is ambiguous or unavailable, provide your best estimate based on the text. " +
                 "Be precise, consistent, and do not include any explanations or extra text—only valid JSON."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=schema,
        )
        return response.output_parsed
