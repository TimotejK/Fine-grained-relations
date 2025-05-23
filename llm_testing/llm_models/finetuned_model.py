import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Set path to the saved model
saved_model_path = "./finetuned_llama/final_model"

# Load 4-bit quantization config (same as during training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(saved_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if needed

# Load base model with quantization config
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ["HF_TOKEN"]
)

# Load LoRA adapters on top of base model
model = PeftModel.from_pretrained(base_model, saved_model_path)

# Set to eval mode
model.eval()

# ---- Example Inference ----
prompt = """Below is a patient discharge summary. Guess how long before or after the admission date did the event marked with <event> tag start and end. Provide your guess as a number of months, days, hours, or minutes.

Text: The patient was admitted on 01/01/2020. <event>The patient developed a rash</event> which worsened over time.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode and print output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Prediction ===")
print(generated_text)
