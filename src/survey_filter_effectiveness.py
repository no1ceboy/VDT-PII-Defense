"""
Survey Privacy Filter Effectiveness (DPO Quality & Standalone Check)
This script evaluates how effective the Privacy Filter is by reading the ALREADY FILTERED
summaries from our generated DPO dataset (`results/dpo_natural_leakage.jsonl` or `results/dpo_dataset.jsonl`)!
"""

import os
import json
from tqdm import tqdm
import argparse

from src.data_loader import DataLoader
from src.evaluate import AttackEvaluator

def main(args):
    print("Loading datasets and initializing Evaluator...")
    loader = DataLoader("datasets")
    all_docs = loader.load_all(["medical"], limit_per_dataset=args.limit)
    doc_map = {doc.id: doc for doc in all_docs}
    
    evaluator = AttackEvaluator()
    
    # ---------------------------------------------------------
    # PART 1: Test Filter on Leaked Summaries (DPO Chosen Quality Check)
    # Automatically check for dpo_natural_leakage.jsonl OR dpo_dataset.jsonl
    # ---------------------------------------------------------
    print("\n--- PART 1: Testing Filter on Existing DPO Chosen Summaries (DPO Quality Check) ---")
    summary_attempts = 0
    summary_leaks_after_filter = 0
    summary_results = []
    
    dpo_files = ["results/dpo_natural_leakage.jsonl", "results/dpo_dataset.jsonl"]
    dpo_file = next((f for f in dpo_files if os.path.exists(f)), None)
    
    if dpo_file:
        print(f"[INFO] Using DPO file: {dpo_file}")
        with open(dpo_file, "r", encoding="utf-8") as f:
            dpo_lines = [json.loads(line) for line in f if line.strip()]
            
        print(f"Found {len(dpo_lines)} DPO pairs to evaluate for residual PII leakage.")
        
        for pair in tqdm(dpo_lines, desc="Evaluating DPO Purity"):
            # Try getting doc_id from metadata or fallback to matching string
            doc_id = pair.get("metadata", {}).get("doc_id")
            doc = doc_map.get(doc_id)
            
            # If doc_id is not stored directly, try finding it by matching reference summary
            if not doc:
                for d in all_docs:
                    if d.reference_summary[:100] in pair["chosen"][0]["content"] or d.id in str(pair.get("metadata", "")):
                        doc = d
                        break
            
            # Default fallback if still not found
            if not doc and all_docs:
                doc = all_docs[0]
                
            rejected_summary = pair["rejected"][0]["content"]
            chosen_summary = pair["chosen"][0]["content"] # Already filtered!
            gold_pii_flat = doc.metadata.get("gold_pii_flat", [])
            
            # Evaluate if any PII survived in the Chosen (filtered) text
            eval_result = evaluator.evaluate(
                attack_category="pii_extraction",
                clean_summary=doc.reference_summary,
                attacked_summary=chosen_summary,
                reference_summary=doc.reference_summary,
                gold_pii=doc.metadata.get("gold_pii"),
                gold_pii_flat=gold_pii_flat,
            )
            
            summary_attempts += 1
            is_leaked = eval_result.attack_success
            if is_leaked:
                summary_leaks_after_filter += 1
                
            summary_results.append({
                "doc_id": getattr(doc, 'id', 'unknown'),
                "original_summary": rejected_summary,
                "redacted_summary": chosen_summary,
                "still_leaked_pii": is_leaked,
                "leak_details": eval_result.details
            })
    else:
        print(f"Error: Neither {dpo_files[0]} nor {dpo_files[1]} was found.")
        
    # ---------------------------------------------------------
    # PART 2: Test Filter on Raw Input Documents (Optional)
    # ---------------------------------------------------------
    raw_attempts = 0
    raw_leaks_after_filter = 0
    raw_results = []
    
    if not args.skip_raw:
        print("\n--- PART 2: Testing Filter on Raw Input Documents (Requires OPF model) ---")
        try:
            from src.openai_privacy_filter import PrivacyFilterDefense
            defense = PrivacyFilterDefense(device="cpu" if args.cpu else "cuda")
            if defense.runtime:
                for doc in tqdm(all_docs, desc="Raw Documents"):
                    raw_text = doc.document
                    gold_pii_flat = doc.metadata.get("gold_pii_flat", [])
                    redacted_text = defense.redact(raw_text)
                    
                    eval_result = evaluator.evaluate(
                        attack_category="pii_extraction",
                        clean_summary=doc.reference_summary,
                        attacked_summary=redacted_text,
                        reference_summary=doc.reference_summary,
                        gold_pii=doc.metadata.get("gold_pii"),
                        gold_pii_flat=gold_pii_flat,
                    )
                    raw_attempts += 1
                    if eval_result.attack_success:
                        raw_leaks_after_filter += 1
                    raw_results.append({
                        "doc_id": doc.id,
                        "still_leaked_pii": eval_result.attack_success,
                        "leak_details": eval_result.details
                    })
            else:
                print("Skipping Part 2: OPF model failed to load.")
        except Exception as e:
            print(f"Skipping Part 2 due to load error: {e}")
            
    # ---------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL PRIVACY FILTER EFFECTIVENESS REPORT")
    print("="*60)
    
    if summary_attempts > 0:
        summary_rate = (summary_leaks_after_filter / summary_attempts) * 100
        print(f"[DPO Chosen Purity Check] Summary Leakage after Filter: {summary_rate:5.2f}% ({summary_leaks_after_filter}/{summary_attempts} summaries still had residual PII)")
        print(f"                          -> Your DPO Chosen dataset is {100 - summary_rate:5.2f}% PURE!")
        
    if raw_attempts > 0:
        raw_rate = (raw_leaks_after_filter / raw_attempts) * 100
        print(f"[Input Firewall Check]    Raw Document Leakage after Filter: {raw_rate:5.2f}% ({raw_leaks_after_filter}/{raw_attempts})")
    print("="*60)
    
    os.makedirs("results", exist_ok=True)
    with open("results/filter_effectiveness_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "dpo_chosen_purity_test": {
                "total_tested": summary_attempts,
                "leaked_after_filter": summary_leaks_after_filter,
                "leakage_rate": (summary_leaks_after_filter / max(summary_attempts, 1)),
                "details": summary_results
            },
            "raw_documents_test": {
                "total_tested": raw_attempts,
                "leaked_after_filter": raw_leaks_after_filter,
                "details": raw_results
            }
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Detailed statistics saved to results/filter_effectiveness_stats.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Survey Privacy Filter Effectiveness")
    parser.add_argument("--limit", type=int, default=100, help="Number of documents to test")
    parser.add_argument("--cpu", action="store_true", help="Force running filter on CPU")
    parser.add_argument("--skip-raw", action="store_true", help="Skip evaluating raw documents to avoid loading OPF model")
    args = parser.parse_args()
    main(args)
