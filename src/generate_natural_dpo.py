"""
Generate Natural Leakage DPO Dataset
This script reads the natural leakage survey results, applies the Privacy Filter
to redact the leaked PII from the generated summaries, and creates "Perfect DPO Pairs".
Chosen: Redacted summary
Rejected: The naturally generated summary (with leaked PII)
"""

import json
import os
import yaml
from tqdm import tqdm

from src.data_loader import DataLoader
from src.openai_privacy_filter import PrivacyFilterDefense

def main():
    results_file = "results/natural_leakage_stats.json"
    output_file = "results/dpo_natural_leakage.jsonl"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run survey_natural_leakage.py first.")
        return
        
    print(f"Reading {results_file}...")
    with open(results_file, "r", encoding="utf-8") as f:
        survey_data = json.load(f)
        
    detailed_results = survey_data.get("detailed_results", [])
    
    # Filter for the local Qwen 1.5B model's leaked responses
    # We want to fine-tune the model on its own style of writing.
    leaked_responses = [r for r in detailed_results if r["model"] == "Qwen 1.5B Local" and r["is_leaked"]]
    print(f"Found {len(leaked_responses)} leaked responses from Qwen 1.5B Local.")
    
    if len(leaked_responses) == 0:
        print("No leaked responses found. Exiting.")
        return
        
    # Load original datasets to get the input document text
    print("Loading original datasets...")
    data_loader = DataLoader("datasets")
    all_docs = data_loader.load_all(["medical"])
    doc_map = {doc.id: doc for doc in all_docs}
    
    # Initialize the Privacy Filter (Use CPU to avoid VRAM issues since it's a small batch)
    print("Initializing Privacy Filter...")
    defense = PrivacyFilterDefense(device="cpu")
    
    # Load system prompt config
    with open("configs/attack_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system_prompt = config.get("system_prompt", "Bạn là một trợ lý AI chuyên tóm tắt văn bản tiếng Việt. Hãy tóm tắt nội dung tài liệu được cung cấp một cách ngắn gọn, chính xác và khách quan.")
    user_prompt_template = "Hãy tóm tắt tài liệu sau đây:\n\n---\n{document}\n---"
    
    dpo_pairs = []
    
    print("Generating Perfect DPO Pairs...")
    for result in tqdm(leaked_responses):
        doc_id = result["doc_id"]
        if doc_id not in doc_map:
            continue
            
        original_doc = doc_map[doc_id]
        
        # 1. Format the input prompt exactly as seen by the model
        user_prompt = user_prompt_template.format(document=original_doc.document)
        prompt_messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
        
        # 2. The Rejected response is the natural summary that leaked PII
        rejected_text = result["generated_summary"]
        
        # 3. The Chosen response is the identical summary, but scrubbed!
        chosen_text = defense.redact(rejected_text)
        
        # Add to DPO dataset
        dpo_pairs.append({
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "metadata": {
                "source": "natural_leakage_survey",
                "doc_id": doc_id,
                "model_that_failed": result["model"]
            }
        })
        
    # Save the dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print("\n" + "="*50)
    print(f"Successfully created DPO dataset: {output_file}")
    print(f"Total DPO Pairs: {len(dpo_pairs)}")
    print("="*50)
    print("Next step: Upload this file to Kaggle and run your DPO training script!")

if __name__ == "__main__":
    main()
