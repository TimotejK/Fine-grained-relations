import os
import re

import wandb
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from data_loaders.dataset import TimelineDataset
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import compute_metrics, store_prediction_for_error_analysis
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

folder = "./data/"
model_id = "Meta-Llama-3.1-8B-finetuned"
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
EOS_TOKEN = None  # Must add EOS_TOKEN


def log_text(line, log_file="llm_finetuning.log"):
    with open(log_file, "a") as log:
        log.write(line)
        log.write("\n")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }

def prepare_model(model_name, max_seq_length):
    global EOS_TOKEN
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    if "gemma" in model_name:
        model = FastLanguageModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # Turn off for just text!
            finetune_language_layers=True,  # Should leave on!
            finetune_attention_modules=True,  # Attention good for GRPO
            finetune_mlp_modules=True,  # Should leave on always!

            r=8,  # Larger = higher accuracy, but might overfit
            lora_alpha=8,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    return model, tokenizer

def prepare_data(model_name, tokenizer):
    df = load_i2b2_absolute_data(test_split=False)
    datasetPT = TimelineDataset(df, use_qa_format=True)

    def dataset_generator(ptDataset):
        for i in range(len(ptDataset)):
            yield ptDataset[i]
    dataset = Dataset.from_generator(lambda: dataset_generator(datasetPT))

    if "gemma" in model_name:
        def convert_to_conversation(sample):
            return {
                "conversations": [
                    {"content": sample["instruction"] + "\n\n\nInput:\n" + sample["input"], "role": "user"},
                    {"content": sample["output"], "role": "assistant"},
                ]
            }

        dataset = dataset.map(convert_to_conversation)
        def apply_chat_template(examples):
            texts = tokenizer.apply_chat_template(examples["conversations"])
            return {"text": texts}

        dataset = dataset.map(apply_chat_template, batched=True)
    else:
        dataset = dataset.map(formatting_prompts_func, batched=True, )
    return dataset

def train_the_model(model, dataset, tokenizer, max_seq_length):
    wandb.init(project="fine-grained-finetuning")
    training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=5,  # Set this for 1 full training run.
            # max_steps = 60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",  # Use this for WandB etc
        )
    if "gemma" in model_id:
        training_args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 5, # Set this for 1 full training run.
            # max_steps = 3,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=training_args,
    )
    if "gemma" in model_id:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
    trainer_stats = trainer.train()



def parse_answer(answer):
    # Regex to extract 'START: ...' and 'END: ...'
    start_match = re.search(r"START:\s*([^\.,\n]+)", answer, re.IGNORECASE)
    end_match = re.search(r"END:\s*([^\.,\n]+)", answer, re.IGNORECASE)
    def to_minutes(text):
        time_units_to_minutes = {
            "minute": 1,
            "hour": 60,
            "day": 1440,
            "month": 43200,
            "year": 1440 * 365
        }
        m = re.match(r"(\d+)\s+(\w+)[s]?\s+(before|after)", text.strip())
        if not m:
            return None
        value, unit, direction = m.groups()
        value = int(value)
        unit = unit.lower()
        minutes = value * time_units_to_minutes.get(unit, 1)
        if direction == "before":
            minutes = -minutes
        return minutes
    start = to_minutes(start_match.group(1)) if start_match else 0
    end = to_minutes(end_match.group(1)) if end_match else 0
    return {"start": start, "end": end}

def test_the_model(model, df, tokenizer):
    datasetPT = TimelineDataset(df, use_qa_format=True)

    def dataset_generator(ptDataset):
        for i in range(len(ptDataset)):
            yield ptDataset[i]

    dataset = Dataset.from_generator(lambda: dataset_generator(datasetPT))
    dataset = dataset.map(formatting_prompts_func, batched=True, )

    FastLanguageModel.for_inference(model)
    gold = []
    predicted = []

    predictions_starts = []
    predictions_ends = []
    predictions_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []

    for example in dataset:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    example["instruction"],  # instruction
                    example["input"],  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
        answer = tokenizer.batch_decode(outputs)[0]
        parsed = parse_answer(answer)
        s = parsed["start"] + example["row"]["admission_date_minutes"]
        predictions_starts.append(s)
        e = parsed["end"] + example["row"]["admission_date_minutes"]
        predictions_ends.append(e)
        d = parsed["end"] - parsed["start"]
        predictions_durations.append(d)
        gs = (example["row"]["start_time_minutes"], example["row"]["start_lower_minutes"], example["row"]["start_upper_minutes"])
        gold_starts.append(gs)
        ge = (example["row"]["end_time_minutes"], example["row"]["end_lower_minutes"], example["row"]["end_upper_minutes"])
        gold_ends.append(ge)
        gd = (example["row"]["duration_minutes"], example["row"]["duration_lower_minutes"], example["row"]["duration_upper_minutes"])
        gold_durations.append(gd)

        store_prediction_for_error_analysis(model_id, example["row"]["document_id"], example["row"]["text"],
                                            example["row"]["event_id"], example["row"]["start_char"],
                                            example["row"]["end_char"], s, e, d, gs, ge, gd)


        eval_metrics_partial = compute_metrics(predictions_starts, predictions_ends, predictions_durations, gold_starts,
                                               gold_ends,
                                               gold_durations)
        print(f"{model_id} --- Partial metrics: {eval_metrics_partial}")
        log_text(f"{model_id} --- Partial metrics: {eval_metrics_partial}", log_file=f"{model_id}_evaluation.log")

        gold.append(example["text"])
        predicted.append(answer)

    eval_metrics = compute_metrics(predictions_starts, predictions_ends, predictions_durations, gold_starts, gold_ends,
                                   gold_durations)
    print(f"{model_id} --- Evaluation metrics: {eval_metrics}")
    log_text(f"{model_id} --- Evaluation metrics: {eval_metrics}", log_file=f"{model_id}_evaluation.log")

    torch.save({"gold": gold, "predicted": predicted}, folder + "finetuned-results-alpaca.pt")

def save_model(model, tokenizer):
    model.save_pretrained(model_id + "-lora_model")  # Local saving
    tokenizer.save_pretrained(model_id + "-lora_model")

def finetune_on_i2b2(model_name):
    global model_card, model_id
    # model_card = "unsloth/Meta-Llama-3.1-8B"
    # model_card = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
    max_seq_length = 4096
    model_card = model_name
    model_id = model_name.split("/")[-1] + "-finetuned"
    model, tokenizer = prepare_model(model_card, max_seq_length)
    dataset = prepare_data(model_card, tokenizer)
    train_the_model(model, dataset, tokenizer, max_seq_length)
    save_model(model, tokenizer)
    df_test = load_i2b2_absolute_data(test_split=True)
    test_the_model(model, df_test, tokenizer)
    pass

if __name__ == '__main__':
    finetune_on_i2b2("unsloth/gemma-3-12b-it-unsloth-bnb-4bit")