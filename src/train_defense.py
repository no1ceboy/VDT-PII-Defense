import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

def main():
    parser = argparse.ArgumentParser(description="Train DPO Defense Model")
    parser.add_argument("--dataset_path", type=str, default="results/dpo_dataset.jsonl", help="Path to the DPO JSONL dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    args = parser.parse_args()

    # Reverting to Qwen2.5 1.5B-Instruct for safer VRAM overhead during DPO on Kaggle
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = args.dataset_path
    
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
        prompt_messages = row["prompt"]
        
        # Aggressively truncate to fit DPO (2x model) in Kaggle T4 VRAM.
        # Keep start+end of document to preserve injected attacks.
        user_content = prompt_messages[1]["content"]
        if len(user_content) > 1500:
            prompt_messages[1]["content"] = user_content[:750] + "\n...[TRUNCATED]...\n" + user_content[-750:]
            
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        # Truncate responses hard to keep total sequence short
        chosen_str = row["chosen"][0]["content"][:500] + tokenizer.eos_token
        rejected_str = row["rejected"][0]["content"][:500] + tokenizer.eos_token
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
        device_map={"": 0}
    )
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # DPOConfig extends TrainingArguments.
    training_args = DPOConfig(
        output_dir="results/defense_model",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name="vdt-pii-defense-dpo",
        beta=args.beta,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print("Starting DPO training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("results/defense_model/final")
    tokenizer.save_pretrained("results/defense_model/final")

if __name__ == "__main__":
    main()

