import os

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, LlamaForCausalLM, \
    LlamaTokenizer
from trl import SFTTrainer
import torch
import pandas as pd
from peft import LoraConfig
import bitsandbytes as bnb

from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data


# --- Time conversion function ---
time_units_to_minutes = {
    "minute": 1,
    "hour": 60,
    "day": 1440,
    "month": 43200,
    "year": 1440 * 365,
}
def convert_time_in_minutes_to_description(time_in_minutes):
    before = False
    if time_in_minutes < 0:
        before = True
        time_in_minutes = abs(time_in_minutes)
    if time_in_minutes > time_units_to_minutes["year"]:
        unit = "year"
        value = time_in_minutes // time_units_to_minutes["year"]
    elif time_in_minutes > time_units_to_minutes["month"]:
        unit = "month"
        value = time_in_minutes // time_units_to_minutes["month"]
    elif time_in_minutes > time_units_to_minutes["day"]:
        unit = "day"
        value = time_in_minutes // time_units_to_minutes["day"]
    elif time_in_minutes > time_units_to_minutes["hour"]:
        unit = "hour"
        value = time_in_minutes // time_units_to_minutes["hour"]
    else:
        unit = "minute"
        value = time_in_minutes
    return f"{value} {unit}{'s' if value > 1 else ''} {'before' if before else 'after'}"

# --- Convert each row to a QA pair ---
def create_conversation(row):
    summary = row["text"]
    summary_with_event = summary[:row["start_char"]] + "<event>" + summary[row["start_char"]:row["end_char"]] + "</event>" + summary[row["end_char"]:]
    prompt = (
        "Below is a patient discharge summary. Guess how long before or after the admission date "
        "did the event marked with <event> tag start and end. Provide your guess as a number of months, days, hours, or minutes.\n\n"
        f"Text: {summary_with_event}"
    )
    answer = (
        f"Start time: {convert_time_in_minutes_to_description(row['start_time_minutes'] - row['admission_date_minutes'])}\n"
        f"End time: {convert_time_in_minutes_to_description(row['end_time_minutes'] - row['admission_date_minutes'])}"
    )
    return pd.Series({"prompt": prompt, "answer": answer})

OUTPUT_DIR = "./finetuned_llama"
def finetune_on_i2b2():
    i2b2_dataset = load_i2b2_absolute_data(test_split=False)

    dataset = i2b2_dataset.apply(create_conversation, axis=1)
    formatted_data = [{"conversations": [{"from": "user", "value": item["prompt"]}, {"from": "gpt", "value": item["answer"]}]} for item in dataset.iloc]
    formatted_dataset = Dataset.from_list(formatted_data)

    # --- Tokenizer & Model ---
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=os.environ["HF_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # Use normalized float 4 for better accuracy
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,   # Apply second quantization for more memory savings
    )

    # Optional: LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,   # Use the 4-bit quantization config
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        token=os.environ["HF_TOKEN"]
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./llama3-i2b2-checkpoints",
        per_device_train_batch_size=2,    # Reduced batch size to help with memory
        gradient_accumulation_steps=4,    # Increased gradient accumulation to compensate
        logging_steps=10,
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=True,                        # Use bfloat16 for training
        logging_dir="./logs",
        optim="paged_adamw_32bit",        # Use 32-bit Adam optimizer
        report_to="none",
        save_total_limit=2,
        gradient_checkpointing=True,      # Enable gradient checkpointing to save memory
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        formatting_func=lambda example: [
            {"role": turn["from"], "content": turn["value"]} for turn in example["conversations"]
        ],
        peft_config=peft_config,
        args=training_args,
    )

    # --- Train! ---
    trainer.train()


    # Save the final model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

    print(f"Model successfully saved to {os.path.join(OUTPUT_DIR, 'final_model')}")

# Example function to demonstrate how to load and use the saved model
def load_and_use_model():
    saved_model_path = os.path.join(OUTPUT_DIR, "final_model")
    loaded_model = LlamaForCausalLM.from_pretrained(saved_model_path)
    loaded_tokenizer = LlamaTokenizer.from_pretrained(saved_model_path)

    # Example usage
    input_text = "Once upon a time"
    inputs = loaded_tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = loaded_model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
        )

    generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

# Uncomment to demonstrate loading and using the model
# load_and_use_model()
if __name__ == '__main__':
    finetune_on_i2b2()