# VDT — Prompt Injection Attack Pipeline for Vietnamese Text Summarization

This pipeline attacks LLMs with prompt injection to study their vulnerability during Vietnamese text summarization. The results will be used to train an RL-based defense agent.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (choose one or both)
set OPENROUTER_API_KEY=your_openrouter_key_here
set GOOGLE_API_KEY=your_google_api_key_here

# Or create a .env file in the VDT/ directory:
# OPENROUTER_API_KEY=sk-or-...
# GOOGLE_API_KEY=AI...
```

## Usage

```bash
# Dry run — see experiment plan without API calls
python -m src.run_attack --dry-run

# Run with default config (10 samples per dataset)
python -m src.run_attack

# Run with fewer samples for quick test
python -m src.run_attack --samples 2 --positions middle

# Custom config
python -m src.run_attack --config configs/attack_config.yaml --datasets-dir datasets
```

## Attack Categories

| Category | Description | Templates |
|---|---|---|
| **Naive Override** | Direct instruction to ignore summarization task | 4 |
| **Summary Biasing** | Bias the summary or inject fabricated information | 3 |
| **PII Extraction** | Extract personal information from documents | 3 |
| **Persona Hijacking** | Trick the model into adopting a different role | 3 |
| **Stealth Injection** | Obfuscated attacks using HTML, Unicode, language switching | 4 |

## Output

Results are saved to `results/`:
- `attack_results.jsonl` — per-experiment results
- `clean_summaries.jsonl` — baseline clean summaries  
- `attack_stats.json` — aggregate statistics (ASR by category/model/difficulty)

## Project Structure

```
VDT/
├── configs/attack_config.yaml    # Configuration
├── datasets/                     # Vietnamese summarization datasets
├── src/
│   ├── data_loader.py            # Unified dataset loader
│   ├── attack_templates.py       # 17 attack templates (5 categories)
│   ├── injector.py               # Document injection engine
│   ├── llm_client.py             # OpenRouter + Google AI Studio client
│   ├── run_attack.py             # Main pipeline
│   └── evaluate.py               # Attack success evaluator
└── results/                      # Output directory
```
