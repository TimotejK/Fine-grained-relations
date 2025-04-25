from ollama import chat
from pydantic import BaseModel


class OllamaModel:
    def __init__(self, model_name='llama3.1'):
        self.model_name = model_name

    def predict(self, prompt):
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=self.model_name
        )
        return response.message.content

    def predict_schema(self, prompt, schema):
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=self.model_name,
            format=schema.model_json_schema(),
        )
        return schema.model_validate_json(response.message.content)


if __name__ == '__main__':
    class Country(BaseModel):
        name: str
        capital: str
        languages: list[str]
    model = OllamaModel(model_name="gemma3:27b")
    pred = model.predict_schema("What is the best country?", schema=Country)
    print(pred)
