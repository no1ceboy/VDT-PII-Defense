"""
Main attack pipeline orchestrator.

Loads config, samples documents, generates clean and attacked summaries,
evaluates attack success, and saves results to JSONL.

Usage:
    python -m src.run_attack                          # full pipeline
    python -m src.run_attack --dry-run                # no API calls
    python -m src.run_attack --samples 3 --positions middle
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .data_loader import DataLoader
from .attack_templates import get_templates, ALL_TEMPLATES
from .injector import DocumentInjector
from .llm_client import LLMClient
from .evaluate import AttackEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_list(config: dict) -> list[dict]:
    """Build a flat list of models with provider info."""
    models = []
    for model_info in config.get("models", {}).get("openrouter", []):
        models.append({
            "id": model_info["id"],
            "name": model_info["name"],
            "provider": "openrouter",
        })
    for model_info in config.get("models", {}).get("google", []):
        models.append({
            "id": model_info["id"],
            "name": model_info["name"],
            "provider": "google",
        })
    return models


def run_pipeline(args):
    """Run the attack pipeline."""

    # ----------------------------------------------------------
    # 1. Load configuration
    # ----------------------------------------------------------
    config = load_config(args.config)
    load_dotenv()  # load .env file if present

    logger.info("=" * 60)
    logger.info("Prompt Injection Attack Pipeline")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # 2. Load and sample documents
    # ----------------------------------------------------------
    datasets_dir = args.datasets_dir
    sampling_cfg = config.get("sampling", {})
    n_samples = args.samples or sampling_cfg.get("samples_per_dataset", 10)
    seed = sampling_cfg.get("random_seed", 42)
    dataset_names = sampling_cfg.get("datasets", None)

    logger.info(f"Loading datasets from: {datasets_dir}")
    loader = DataLoader(datasets_dir)
    # We load 3x the required samples to give the random sampler enough variety,
    # but strictly limit it so it doesn't parse 150,000 files.
    all_docs = loader.load_all(dataset_names, limit_per_dataset=n_samples * 3)
    sampled_docs = loader.sample(all_docs, n_per_dataset=n_samples, seed=seed)
    logger.info(f"Sampled {len(sampled_docs)} documents for attack experiments")

    # ----------------------------------------------------------
    # 3. Set up attack components
    # ----------------------------------------------------------
    attack_cfg = config.get("attack", {})
    positions = args.positions.split(",") if args.positions else attack_cfg.get("injection_positions", ["middle"])
    categories = attack_cfg.get("categories", None)

    templates = get_templates(categories=categories)
    
    if getattr(args, "fast", False):
        fast_templates = []
        seen_cats = set()
        for t in templates:
            if t.category not in seen_cats:
                fast_templates.append(t)
                seen_cats.add(t.category)
        templates = fast_templates
        logger.info(f"FAST MODE: Reduced to {len(templates)} templates (1 per category)")

    logger.info(f"Using {len(templates)} attack templates across {len(set(t.category for t in templates))} categories")
    logger.info(f"Injection positions: {positions}")

    injector = DocumentInjector(seed=seed)
    evaluator = AttackEvaluator()

    # ----------------------------------------------------------
    # 4. Set up LLM client
    # ----------------------------------------------------------
    llm_cfg = config.get("llm", {})
    models = build_model_list(config)
    logger.info(f"Target models: {[m['name'] for m in models]}")

    client = LLMClient(
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 1024),
        max_retries=llm_cfg.get("max_retries", 3),
        retry_delay=llm_cfg.get("retry_delay_seconds", 5),
        request_delay=llm_cfg.get("request_delay_seconds", 2),
    )

    system_prompt = config.get("system_prompt", "Tóm tắt văn bản sau.")
    user_prompt_template = config.get("user_prompt_template", "Hãy tóm tắt:\n\n{document}")

    # ----------------------------------------------------------
    # 5. Prepare output
    # ----------------------------------------------------------
    output_cfg = config.get("output", {})
    results_dir = Path(output_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / output_cfg.get("results_file", "attack_results.jsonl")
    clean_file = results_dir / output_cfg.get("clean_summaries_file", "clean_summaries.jsonl")

    # ----------------------------------------------------------
    # 6. Run experiments
    # ----------------------------------------------------------
    total_experiments = len(sampled_docs) * len(templates) * len(positions) * len(models)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"  = {len(sampled_docs)} docs × {len(templates)} templates × {len(positions)} positions × {len(models)} models")

    if args.dry_run:
        logger.info("[DRY RUN] Skipping API calls. Showing experiment plan:")
        for doc in sampled_docs[:3]:
            logger.info(f"  Doc: [{doc.source_dataset}] {doc.id}")
        for t in templates[:3]:
            logger.info(f"  Template: [{t.category}] {t.name}")
        for m in models:
            logger.info(f"  Model: {m['name']} ({m['provider']})")
        logger.info(f"  Positions: {positions}")
        logger.info(f"\n[DRY RUN] Would generate {total_experiments} experiments. Exiting.")
        return

    # Cache clean summaries to avoid redundant API calls
    clean_cache = {}  # key: (doc.id, model_id) -> clean_summary
    if clean_file.exists():
        with open(clean_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    clean_cache[(record["doc_id"], record["model"])] = record.get("clean_summary", "")
                except Exception:
                    pass
        logger.info(f"Loaded {len(clean_cache)} cached clean summaries.")

    completed_attacks = set()
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    key = (record["doc_id"], record["model"], record["attack_template_id"], record["injection_position"])
                    completed_attacks.add(key)
                except Exception:
                    pass
        logger.info(f"Found {len(completed_attacks)} previously completed attacks. Resuming...")

    all_results = []
    progress = tqdm(total=total_experiments, desc="Running attacks")

    for doc in sampled_docs:
        for model in models:
            model_id = model["id"]
            provider = model["provider"]

            # --- Get clean summary (baseline) ---
            cache_key = (doc.id, model_id)
            if cache_key not in clean_cache:
                logger.info(f"Getting clean summary: {doc.id} → {model['name']}")
                clean_resp = client.summarize(
                    document=doc.document,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    model_id=model_id,
                    provider=provider,
                )
                clean_cache[cache_key] = clean_resp.output

                # Save clean summary
                clean_record = {
                    "doc_id": doc.id,
                    "source_dataset": doc.source_dataset,
                    "model": model_id,
                    "clean_summary": clean_resp.output,
                    "error": clean_resp.error,
                    "latency": clean_resp.latency_seconds,
                }
                with open(clean_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

            clean_summary = clean_cache[cache_key]

            # --- Run attacks ---
            for template in templates:
                for position in positions:
                    attack_key = (doc.id, model_id, template.id, position)
                    if attack_key in completed_attacks:
                        progress.update(1)
                        continue
                        
                    # Inject
                    injection = injector.inject(doc.document, template, position)

                    # Get attacked summary
                    logger.debug(f"Attack: {doc.id} × {template.id} × {position} → {model['name']}")
                    attacked_resp = client.summarize(
                        document=injection.poisoned_document,
                        system_prompt=system_prompt,
                        user_prompt_template=user_prompt_template,
                        model_id=model_id,
                        provider=provider,
                    )

                    # Evaluate
                    eval_result = evaluator.evaluate(
                        attack_category=template.category,
                        clean_summary=clean_summary,
                        attacked_summary=attacked_resp.output,
                        reference_summary=doc.reference_summary,
                        gold_pii=doc.metadata.get("gold_pii"),
                        gold_pii_flat=doc.metadata.get("gold_pii_flat"),
                    )

                    # Build result record
                    record = {
                        "doc_id": doc.id,
                        "source_dataset": doc.source_dataset,
                        "domain": doc.domain,
                        "model": model_id,
                        "model_name": model["name"],
                        "provider": provider,
                        "attack_template_id": template.id,
                        "attack_category": template.category,
                        "attack_difficulty": template.difficulty,
                        "attack_name": template.name,
                        "injection_position": position,
                        "reference_summary": doc.reference_summary[:500],
                        "clean_summary": clean_summary[:500],
                        "attacked_summary": attacked_resp.output[:500],
                        "attack_success": eval_result.attack_success,
                        "attack_success_score": eval_result.attack_success_score,
                        "evaluation_method": eval_result.evaluation_method,
                        "evaluation_details": eval_result.details,
                        "error": attacked_resp.error,
                        "latency": attacked_resp.latency_seconds,
                    }

                    all_results.append(record)

                    # Append to JSONL (streaming write for crash resilience)
                    with open(results_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    progress.update(1)

    progress.close()

    # ----------------------------------------------------------
    # 7. Print summary statistics
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("ATTACK RESULTS SUMMARY")
    logger.info("=" * 60)

    stats = AttackEvaluator.compute_asr(all_results)

    overall = stats["overall"]
    logger.info(f"Overall ASR: {overall['success']}/{overall['total']} = {overall['asr']:.1%}")

    logger.info("\nBy Category:")
    for cat, s in stats["by_category"].items():
        logger.info(f"  {cat}: {s['success']}/{s['total']} = {s['asr']:.1%}")

    logger.info("\nBy Model:")
    for model, s in stats["by_model"].items():
        logger.info(f"  {model}: {s['success']}/{s['total']} = {s['asr']:.1%}")

    logger.info("\nBy Difficulty:")
    for diff, s in stats["by_difficulty"].items():
        logger.info(f"  {diff}: {s['success']}/{s['total']} = {s['asr']:.1%}")

    # Save stats
    stats_file = results_dir / "attack_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Prompt Injection Attack Pipeline")
    parser.add_argument(
        "--config",
        default="configs/attack_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment plan without making API calls",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override samples_per_dataset from config",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Comma-separated injection positions (e.g., 'middle,end')",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only 1 template per category for faster testing",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
