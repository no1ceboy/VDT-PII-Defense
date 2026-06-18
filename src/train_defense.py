import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

def main():
    # Use Qwen2.5 1.5B because it has excellent Vietnamese capability and fits locally
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "results/dpo_dataset.jsonl"
    
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Split into train/eval (10% for eval)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # We format the dataset using the model's chat template
    def format_dpo_row(row):
        prompt_str = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        # chosen and rejected are list of dicts: [{"role": "assistant", "content": ...}]
        chosen_str = row["chosen"][0]["content"] + tokenizer.eos_token
        rejected_str = row["rejected"][0]["content"] + tokenizer.eos_token
        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str
        }

    print("Formatting dataset...")
    train_dataset = train_dataset.map(format_dpo_row, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_dpo_row, remove_columns=eval_dataset.column_names)

    print("Loading model with 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Adjust batch size and max length based on VRAM limits.
    # These settings target ~8-12GB VRAM.
    training_args = DPOConfig(
        output_dir="results/defense_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        max_length=2048,
        max_prompt_length=1500,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        beta=0.1, # KL penalty
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    
    print("Starting DPO training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("results/defense_model/final")
    tokenizer.save_pretrained("results/defense_model/final")

if __name__ == "__main__":
    main()
