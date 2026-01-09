"""
RAG Fine-tuning Pipeline for Gemma 3 12B
This module handles fine-tuning the Gemma 3 12B model for RAG-based temporal extraction
"""

import unsloth
import json
import torch
import wandb
from datetime import datetime
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data

class RAGFineTuner:
    def __init__(self, model_name="unsloth/gemma-3-12b-it-unsloth-bnb-4bit", max_seq_length=2048):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def prepare_model(self):
        """Prepare the Gemma 3 12B model for fine-tuning"""
        print(f"Loading model: {self.model_name} (max_seq_length={self.max_seq_length})")

        # Use None for dtype to let unsloth choose the best dtype automatically
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Let unsloth choose the best dtype (will use float16 or bfloat16)
            load_in_4bit=True,
        )

        # Configure LoRA adaptation with smaller rank to save memory
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=8,  # Reduced LoRA rank from 16 to 8 to save memory
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            random_state=3407,
        )

        # Enable gradient checkpointing to save memory during training
        try:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except Exception as e:
            print(f"Note: Could not enable gradient checkpointing: {e}")

        # Set up chat template for Gemma
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma-3",
        )

        print("Model prepared for fine-tuning")

    def load_rag_dataset(self, dataset_path='rag_llm_approach/rag_finetuning_dataset.json', use_streaming=True):
        """Load the RAG fine-tuning dataset with optional streaming (lazy tokenization)"""
        print(f"Loading dataset from: {dataset_path}")

        try:
            with open(dataset_path, 'r') as f:
                qa_pairs = json.load(f)
        except FileNotFoundError:
            print("Dataset not found. Generating new dataset...")
            # Import and call the dataset generation function
            from rag_absolute_time_extraction import generate_finetuning_dataset
            qa_pairs = generate_finetuning_dataset()

        print(f"Loading {len(qa_pairs)} examples...")

        # Convert to the format expected by SFTTrainer
        # Apply chat template immediately but keep it as strings (not tokenized)
        formatted_data = []

        print("Applying chat templates (this will NOT tokenize, just format as strings)...")
        for idx, qa in enumerate(qa_pairs):
            # Create conversation structure
            conversation = [
                {
                    "role": "user",
                    "content": qa["instruction"] + "\n\n" + qa["input"]
                },
                {
                    "role": "assistant",
                    "content": qa["output"]
                }
            ]

            # Apply chat template to get formatted string
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,  # Don't tokenize, just format as string
                add_generation_prompt=False
            )

            # Store as dict with "text" field (expected by SFTTrainer)
            formatted_data.append({"text": text})

            if (idx + 1) % 1000 == 0:
                print(f"Formatted {idx + 1}/{len(qa_pairs)} examples...")

        # Free memory
        del qa_pairs

        # Create dataset with "text" field
        self.dataset = Dataset.from_list(formatted_data)

        # Free memory
        del formatted_data

        print(f"Dataset loaded with {len(self.dataset)} examples")
        print("Chat templates applied - dataset contains formatted strings (not tokenized yet)")
        print("Tokenization will happen during training by SFTTrainer")


    def train(self, output_dir="./rag_finetuned_model", epochs=3, batch_size=1, use_streaming=True):
        """Fine-tune the model"""
        print("Starting fine-tuning...")

        # Initialize wandb
        wandb.init(project="rag-temporal-finetuning", name=f"rag-gemma-3-12b-{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Training configuration optimized for memory efficiency
        training_config = SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=500,
            save_total_limit=2,
            report_to="wandb",
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            # Use single process for dataset processing
            dataset_num_proc=1,
            # Use the "text" field from our dataset
            dataset_text_field="text",
        )

        print("Creating SFTTrainer...")

        # Create trainer - will tokenize the "text" field on-the-fly during training
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            args=training_config,
            max_seq_length=self.max_seq_length,
            packing=False,  # Disable packing to avoid issues with variable length sequences
        )

        # Train only on model responses
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        print("Starting training - tokenization will happen on-the-fly during training")

        # Start training
        trainer_stats = trainer.train()

        print(f"Training completed. Stats: {trainer_stats}")

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to: {output_dir}")

        wandb.finish()

    def evaluate_on_test_set(self, model_dir="./rag_finetuned_model"):
        """Evaluate the fine-tuned model on test set"""
        print("Evaluating fine-tuned model...")

        # Load fine-tuned model
        eval_model, eval_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(eval_model)

        # Load test data
        df_test = load_i2b2_absolute_data(test_split=True)

        # Run evaluation using the RAG extractor
        from rag_absolute_time_extraction import RAGTemporalExtractor

        # Create evaluator with fine-tuned model
        evaluator = RAGTemporalExtractor(use_finetuned=True)
        evaluator.model = eval_model
        evaluator.tokenizer = eval_tokenizer

        # Run inference on test set
        results = []
        correct_predictions = 0
        total_predictions = 0

        for idx, row in df_test.iterrows():
            try:
                event_string = row['text'][row['start_char']:row['end_char']]

                # Create prompt (simplified for evaluation)
                prompt = f"Extract temporal information for this clinical event: {event_string}\nContext: {row['text'][:500]}...\nReturn JSON with start and end times."

                prediction = evaluator.call_llm(prompt)

                results.append({
                    'document_id': row['document_id'],
                    'event_text': event_string,
                    'prediction': prediction,
                    'ground_truth': {
                        'start_time_minutes': row['start_time_minutes'],
                        'end_time_minutes': row['end_time_minutes']
                    }
                })

                total_predictions += 1

                # Simple accuracy check (can be made more sophisticated)
                if prediction.get('start') and prediction.get('end'):
                    correct_predictions += 1

                if idx % 50 == 0:
                    print(f"Evaluated {idx}/{len(df_test)} examples")

            except Exception as e:
                print(f"Error evaluating example {idx}: {e}")
                continue

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Evaluation completed. Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")

        # Save evaluation results
        eval_output = f"rag_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation results saved to: {eval_output}")

        return results

def main(args = None):
    if args is None:
        import argparse

        parser = argparse.ArgumentParser(description='RAG Fine-tuning Pipeline')
        parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                           help='Mode: train or evaluate')
        parser.add_argument('--model_name', default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
                           help='Base model name')
        parser.add_argument('--output_dir', default="./rag_finetuned_model",
                           help='Output directory for fine-tuned model')
        parser.add_argument('--epochs', type=int, default=3,
                           help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=1,
                           help='Training batch size (default 1 for 16GB VRAM)')
        parser.add_argument('--max_seq_length', type=int, default=2048,
                           help='Maximum sequence length (reduce to save memory, default 2048)')
        parser.add_argument('--dataset_path', default='rag_llm_approach/rag_finetuning_dataset.json',
                           help='Path to fine-tuning dataset')

        args = parser.parse_args()

    # Initialize fine-tuner with specified max_seq_length
    finetuner = RAGFineTuner(model_name=args.model_name, max_seq_length=args.max_seq_length)

    if args.mode == 'train':
        # Training pipeline
        finetuner.prepare_model()
        finetuner.load_rag_dataset(args.dataset_path, use_streaming=True)  # Always use streaming
        finetuner.train(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Optional: run evaluation after training
        print("\nRunning post-training evaluation...")
        finetuner.evaluate_on_test_set(args.output_dir)

    elif args.mode == 'evaluate':
        # Evaluation only
        finetuner.evaluate_on_test_set(args.output_dir)

if __name__ == '__main__':
    main()
