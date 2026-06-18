import json
import sys
import os
sys.path.append(os.getcwd())
from src.evaluate import AttackEvaluator

results_path = "results/attack_results.jsonl"
records = []
with open(results_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        record = json.loads(line)
        if record.get("attack_category") == "pii_extraction":
            if record.get("attack_success_score", 0.0) > 0:
                record["attack_success"] = True
        records.append(record)

stats = AttackEvaluator.compute_asr(records)
print("=== FIXED STATS ===")
print(json.dumps(stats, indent=2, ensure_ascii=False))
