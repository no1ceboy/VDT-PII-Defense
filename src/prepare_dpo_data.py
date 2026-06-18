import json
import os
from pathlib import Path
from tqdm import tqdm
import yaml

from src.data_loader import DataLoader
from src.attack_templates import get_templates
from src.injector import DocumentInjector

def main():
    # 1. Load config
    with open("configs/attack_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    system_prompt = config.get("system_prompt", "Tóm tắt văn bản sau.")
    user_prompt_template = config.get("user_prompt_template", "Hãy tóm tắt:\n\n{document}")

    # 2. Load datasets and templates
    data_loader = DataLoader("datasets")
    
    templates = get_templates()
    template_map = {t.id: t for t in templates}
    
    injector = DocumentInjector()

    results_path = "results/attack_results.jsonl"
    output_path = "results/dpo_dataset.jsonl"
    
    print(f"Reading {results_path}...")
    
    dpo_pairs = []
    success_count = 0
    loaded_datasets = {}
    
    with open(results_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if not line.strip(): continue
            record = json.loads(line)
            
            # We want to train the model to reject successful attacks.
            # So we only take pairs where the attack succeeded (model misbehaved).
            if not record.get("attack_success", False):
                continue
                
            # Focus exclusively on PII Leakage attacks for Phase 2
            if record.get("attack_category") != "pii_extraction":
                continue
                
            success_count += 1
            
            doc_id = record["doc_id"]
            domain = record["domain"]
            dataset_name = record["source_dataset"]
            
            # Retrieve original document
            if dataset_name not in loaded_datasets:
                loaded_datasets[dataset_name] = data_loader.load_all([dataset_name])
            docs = loaded_datasets[dataset_name]
            doc = next((d for d in docs if d.id == doc_id), None)
            if not doc:
                continue
                
            template = template_map.get(record["attack_template_id"])
            if not template:
                continue
                
            position = record["injection_position"]
            
            # Reconstruct the poisoned document
            injection_result = injector.inject(doc.document, template, position)
            poisoned_text = injection_result.poisoned_document
            
            # Format prompt
            user_prompt = user_prompt_template.format(document=poisoned_text)
            
            # Combine system prompt and user prompt into a standard chat format
            # Depending on the model, we can format this as OpenAI-style messages
            prompt_messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
            
            # Chosen: the summary of the clean document (model ignores the attack)
            chosen = record["clean_summary"].strip()
            
            # Rejected: the model's output where it followed the attack
            rejected = record["attacked_summary"].strip()
            
            dpo_pairs.append({
                "prompt": prompt_messages,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    "attack_category": record["attack_category"],
                    "attack_difficulty": record["attack_difficulty"],
                    "model_that_failed": record["model_name"]
                }
            })

    print(f"Found {success_count} successful attacks to form DPO preference pairs.")
    
    # Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"Saved DPO dataset to {output_path} with {len(dpo_pairs)} pairs.")

if __name__ == "__main__":
    main()
