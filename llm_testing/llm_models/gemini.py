import os
import hashlib
import json
from google import genai

class GeminiModel:
    def __init__(self, api_key, cache_dir="cache_gemini"):
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)  # Ensure the cache directory exists

    def _get_cache_file_path(self, prompt):
        # Create a unique hash for the prompt to use as a filename
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

    def predict(self, prompt):
        # Get cache path for the given prompt
        cache_path = self._get_cache_file_path(prompt)

        # Attempt to load from cache
        cached_response = self._load_from_cache(cache_path)
        if cached_response:
            print("Loaded from cache.")
            return cached_response

        # If not in cache, make API call
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )

        # Cache and return the response
        response_text = response.text  # Assuming this is a string
        self._save_to_cache(cache_path, response_text)
        return response_text

    def predict_schema(self, prompt, schema):
        # Get cache path for the schema + prompt combination
        cache_path = self._get_cache_file_path(prompt + schema.__name__)

        # Attempt to load from cache
        cached_response = self._load_from_cache(cache_path)
        if cached_response:
            print("Loaded from cache.")
            return cached_response

        # If not in cache, make API call
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )

        # Cache and return the response
        response_parsed = response.parsed  # Assuming this is JSON-serializable
        self._save_to_cache(cache_path, response_parsed)
        return response_parsed
