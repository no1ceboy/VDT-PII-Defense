"""
Survey Privacy Filter Effectiveness (Standalone Defense & DPO Quality Check)
This script evaluates how effective the OpenAI Privacy Filter is by itself:
1. Input Firewall Test: How much PII remains when filtering raw medical documents?
2. Output Firewall Test (DPO Quality Check): How much PII remains when scrubbing the naturally leaked Qwen 1.5B summaries?
"""

import os
import json
import torch
from tqdm import tqdm
import argparse

from src.data_loader import DataLoader
from src.evaluate import AttackEvaluator
from src.openai_privacy_filter import PrivacyFilterDefense

def main(args):
    print("Loading datasets and initializing Privacy Filter...")
    loader = DataLoader("datasets")
    all_docs = loader.load_all(["medical"], limit_per_dataset=args.limit)
    doc_map = {doc.id: doc for doc in all_docs}
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Initializing PrivacyFilterDefense on {device}...")
    defense = PrivacyFilterDefense(device=device)
    evaluator = AttackEvaluator()
    
    # ---------------------------------------------------------
    # PART 1: Test Filter on Raw Input Documents (Input Firewall)
    # ---------------------------------------------------------
    print("\n--- PART 1: Testing Filter on Raw Input Documents ---")
    raw_attempts = 0
    raw_leaks_after_filter = 0
    raw_results = []
    
    for doc in tqdm(all_docs, desc="Raw Documents"):
        raw_text = doc.document
        gold_pii_flat = doc.metadata.get("gold_pii_flat", [])
        
        # Apply filter
        redacted_text = defense.redact(raw_text)
        
        # Check if any PII survived the filter
        eval_result = evaluator.evaluate(
            attack_category="pii_extraction",
            clean_summary=doc.reference_summary,
            attacked_summary=redacted_text,
            reference_summary=doc.reference_summary,
            gold_pii=doc.metadata.get("gold_pii"),
            gold_pii_flat=gold_pii_flat,
        )
        
        raw_attempts += 1
        is_leaked = eval_result.attack_success
        if is_leaked:
            raw_leaks_after_filter += 1
            
        raw_results.append({
            "doc_id": doc.id,
            "original_length": len(raw_text),
            "redacted_sample": redacted_text[:300] + "...",
            "still_leaked_pii": is_leaked,
            "leak_details": eval_result.details
        })
        
    # ---------------------------------------------------------
    # PART 2: Test Filter on Leaked Summaries (DPO Chosen Quality Check)
    # ---------------------------------------------------------
    print("\n--- PART 2: Testing Filter on Leaked Qwen 1.5B Summaries (DPO Quality Check) ---")
    summary_attempts = 0
    summary_leaks_after_filter = 0
    summary_results = []
    
    results_file = "results/natural_leakage_stats.json"
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            survey_data = json.load(f)
            
        leaked_responses = [r for r in survey_data.get("detailed_results", []) if r["model"] == "Qwen 1.5B Local" and r["is_leaked"]]
        print(f"Found {len(leaked_responses)} leaked Qwen summaries to evaluate.")
        
        for result in tqdm(leaked_responses, desc="Leaked Summaries"):
            doc_id = result["doc_id"]
            if doc_id not in doc_map:
                continue
                
            doc = doc_map[doc_id]
            leaked_summary = result["generated_summary"]
            gold_pii_flat = doc.metadata.get("gold_pii_flat", [])
            
            # Apply filter to the summary (This is what becomes the 'Chosen' text in DPO!)
            redacted_summary = defense.redact(leaked_summary)
            
            eval_result = evaluator.evaluate(
                attack_category="pii_extraction",
                clean_summary=doc.reference_summary,
                attacked_summary=redacted_summary,
                reference_summary=doc.reference_summary,
                gold_pii=doc.metadata.get("gold_pii"),
                gold_pii_flat=gold_pii_flat,
            )
            
            summary_attempts += 1
            is_leaked = eval_result.attack_success
            if is_leaked:
                summary_leaks_after_filter += 1
                
            summary_results.append({
                "doc_id": doc_id,
                "original_summary": leaked_summary,
                "redacted_summary": redacted_summary,
                "still_leaked_pii": is_leaked,
                "leak_details": eval_result.details
            })
    else:
        print(f"Warning: {results_file} not found. Skipping Part 2.")
        
    # ---------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL PRIVACY FILTER EFFECTIVENESS REPORT")
    print("="*60)
    
    if raw_attempts > 0:
        raw_rate = (raw_leaks_after_filter / raw_attempts) * 100
        print(f"[Input Firewall] Raw Document Leakage after Filter: {raw_rate:5.2f}% ({raw_leaks_after_filter}/{raw_attempts} documents still had residual PII)")
        print(f"                 -> The Privacy Filter alone eliminated {100 - raw_rate:5.2f}% of document-level PII exposure.")
        
    if summary_attempts > 0:
        summary_rate = (summary_leaks_after_filter / summary_attempts) * 100
        print(f"[Output / DPO Check] Summary Leakage after Filter:    {summary_rate:5.2f}% ({summary_leaks_after_filter}/{summary_attempts} summaries still had residual PII)")
        print(f"                 -> DPO Chosen dataset purity is {100 - summary_rate:5.2f}%!")
    print("="*60)
    
    os.makedirs("results", exist_ok=True)
    with open("results/filter_effectiveness_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "raw_documents_test": {
                "total_tested": raw_attempts,
                "leaked_after_filter": raw_leaks_after_filter,
                "leakage_rate": (raw_leaks_after_filter / max(raw_attempts, 1)),
                "details": raw_results
            },
            "leaked_summaries_test": {
                "total_tested": summary_attempts,
                "leaked_after_filter": summary_leaks_after_filter,
                "leakage_rate": (summary_leaks_after_filter / max(summary_attempts, 1)),
                "details": summary_results
            }
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Detailed statistics saved to results/filter_effectiveness_stats.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Survey Privacy Filter Effectiveness")
    parser.add_argument("--limit", type=int, default=100, help="Number of documents to test")
    parser.add_argument("--cpu", action="store_true", help="Force running filter on CPU")
    args = parser.parse_args()
    main(args)
