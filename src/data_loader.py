"""
Unified data loader for all Vietnamese summarization datasets.

Loads documents from vietnews, VLSP, WikiLingua, ViMs, legal, and medical
datasets into a common schema for the attack pipeline.
"""

import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """Common schema for a document across all datasets."""
    id: str
    source_dataset: str                # vietnews, vlsp, wikilingua, vims, legal, medical
    domain: str                        # news, legal, medical, howto
    document: str                      # full document text
    reference_summary: str             # ground-truth summary
    metadata: dict = field(default_factory=dict)  # dataset-specific (e.g., gold_pii)

    def to_dict(self) -> dict:
        return asdict(self)


class DataLoader:
    """Unified loader for all 6 Vietnamese summarization datasets."""

    def __init__(self, datasets_dir: str):
        self.datasets_dir = Path(datasets_dir)

    def load_all(self, dataset_names: Optional[List[str]] = None, limit_per_dataset: Optional[int] = None) -> List[Document]:
        """Load documents from specified datasets (or all if None)."""
        all_datasets = ["vietnews", "vlsp", "wikilingua", "legal", "medical"]
        names = dataset_names or all_datasets

        documents = []
        for name in names:
            loader = getattr(self, f"_load_{name}", None)
            if loader is None:
                print(f"[WARN] No loader for dataset '{name}', skipping.")
                continue
            try:
                docs = loader(limit=limit_per_dataset)
                print(f"[INFO] Loaded {len(docs)} documents from '{name}'")
                documents.extend(docs)
            except Exception as e:
                print(f"[ERROR] Failed to load '{name}': {e}")

        return documents

    def sample(
        self,
        documents: List[Document],
        n_per_dataset: int = 10,
        seed: int = 42,
    ) -> List[Document]:
        """Sample n documents per dataset."""
        rng = random.Random(seed)
        by_dataset = {}
        for doc in documents:
            by_dataset.setdefault(doc.source_dataset, []).append(doc)

        sampled = []
        for dataset_name, docs in by_dataset.items():
            n = min(n_per_dataset, len(docs))
            # If we loaded limited docs, just take them up to n
            sampled.extend(rng.sample(docs, n))
            print(f"[INFO] Sampled {n} from '{dataset_name}'")

        return sampled

    # ------------------------------------------------------------------
    # Per-dataset loaders
    # ------------------------------------------------------------------

    def _load_vietnews(self, limit: Optional[int] = None) -> List[Document]:
        """Load vietnews tokenized articles. Format: line 1 = title, rest = body."""
        data_dir = self.datasets_dir / "vietnews-master" / "vietnews-master" / "data"
        documents = []

        for split in ["train_tokenized", "val_tokenized", "test_tokenized"]:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            
            # Use os.scandir for very fast directory iteration without loading all files
            for entry in os.scandir(split_dir):
                if not entry.is_file() or not entry.name.endswith(".txt.seg"):
                    continue

                fpath = Path(entry.path)
                try:
                    text = fpath.read_text(encoding="utf-8").strip()
                except UnicodeDecodeError:
                    continue
                    
                if not text:
                    continue
                lines = text.split("\n")
                title = lines[0].strip()
                body_lines = [l.strip() for l in lines[1:] if l.strip()]
                body = "\n".join(body_lines)

                if not body:
                    continue

                documents.append(Document(
                    id=f"vietnews_{split}_{fpath.stem}",
                    source_dataset="vietnews",
                    domain="news",
                    document=body,
                    reference_summary=title,
                    metadata={"split": split, "filename": fpath.name},
                ))
                
                if limit is not None and len(documents) >= limit:
                    return documents

        return documents

    def _load_vlsp(self, limit: Optional[int] = None) -> List[Document]:
        """Load VLSP multi-document summarization dataset (JSONL)."""
        vlsp_dir = self.datasets_dir / "vlsp" / "vlsp"
        documents = []

        for split_file in ["train.label.jsonl", "val.label.jsonl", "test.label.jsonl"]:
            fpath = vlsp_dir / split_file
            if not fpath.exists():
                continue
            split_name = split_file.split(".")[0]

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    item_id = item.get("id", "unknown")

                    text_clusters = item.get("text", [])
                    all_text = []
                    for cluster in text_clusters:
                        if isinstance(cluster, list):
                            all_text.extend(cluster)
                        else:
                            all_text.append(str(cluster))
                    full_doc = "\n".join(all_text)

                    summary_parts = item.get("summary", [])
                    if isinstance(summary_parts, list):
                        summary = "\n".join(summary_parts)
                    else:
                        summary = str(summary_parts)

                    if not full_doc or not summary:
                        continue

                    if len(full_doc) > 3000:
                        full_doc = full_doc[:3000] + "..."

                    documents.append(Document(
                        id=f"vlsp_{split_name}_{item_id}",
                        source_dataset="vlsp",
                        domain="news",
                        document=full_doc,
                        reference_summary=summary,
                        metadata={"split": split_name, "label": item.get("label")},
                    ))
                    
                    if limit is not None and len(documents) >= limit:
                        return documents

        return documents

    def _load_wikilingua(self, limit: Optional[int] = None) -> List[Document]:
        """Load WikiLingua Vietnamese how-to dataset (JSON with src/tgt)."""
        wiki_dir = self.datasets_dir / "wikilingua" / "wikilingua"
        documents = []

        for split_file in ["train.json", "val.json", "test.json"]:
            fpath = wiki_dir / split_file
            if not fpath.exists():
                continue
            split_name = split_file.split(".")[0]

            with open(fpath, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)

                    src_sentences = item.get("src", [])
                    tgt_sentences = item.get("tgt", [])

                    full_doc = "\n".join(src_sentences)
                    summary = "\n".join(tgt_sentences)

                    if not full_doc or not summary:
                        continue

                    if len(full_doc) > 3000:
                        full_doc = full_doc[:3000] + "..."

                    documents.append(Document(
                        id=f"wikilingua_{split_name}_{idx}",
                        source_dataset="wikilingua",
                        domain="howto",
                        document=full_doc,
                        reference_summary=summary,
                        metadata={"split": split_name},
                    ))
                    
                    if limit is not None and len(documents) >= limit:
                        return documents

        return documents

    def _load_legal(self, limit: Optional[int] = None) -> List[Document]:
        """Load legal summarization dataset (JSONL with content/summary)."""
        fpath = self.datasets_dir / "legal_100_summarization.jsonl"
        documents = []

        if not fpath.exists():
            return documents

        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                content = item.get("content", "")
                summary = item.get("summary", "")

                if not content or not summary:
                    continue

                if len(content) > 3000:
                    content = content[:3000] + "..."

                documents.append(Document(
                    id=f"legal_{idx}",
                    source_dataset="legal",
                    domain="legal",
                    document=content,
                    reference_summary=summary,
                    metadata={},
                ))
                
                if limit is not None and len(documents) >= limit:
                    return documents

        return documents

    def _load_medical(self, limit: Optional[int] = None) -> List[Document]:
        """Load medical dataset (JSONL with document/summary/gold_pii)."""
        fpath = self.datasets_dir / "medical_clean_150.jsonl"
        documents = []

        if not fpath.exists():
            return documents

        with open(fpath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                doc_text = item.get("document", "")
                summary = item.get("summary", "")
                gold_pii = item.get("gold_pii", {})
                gold_pii_flat = item.get("gold_pii_flat", [])
                doc_type = item.get("document_type", "")

                if not doc_text or not summary:
                    continue

                if len(doc_text) > 3000:
                    doc_text = doc_text[:3000] + "..."

                documents.append(Document(
                    id=item.get("id", f"medical_{idx}"),
                    source_dataset="medical",
                    domain="medical",
                    document=doc_text,
                    reference_summary=summary,
                    metadata={
                        "gold_pii": gold_pii,
                        "gold_pii_flat": gold_pii_flat,
                        "document_type": doc_type,
                    },
                ))
                
                if limit is not None and len(documents) >= limit:
                    return documents

        return documents

# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Quick fix for Windows unicode print issue
    if sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    datasets_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    loader = DataLoader(datasets_dir)
    docs = loader.load_all(limit_per_dataset=5)
    print(f"\nTotal documents loaded: {len(docs)}")

    for ds in set(d.source_dataset for d in docs):
        count = sum(1 for d in docs if d.source_dataset == ds)
        print(f"  {ds}: {count}")

    sampled = loader.sample(docs, n_per_dataset=3, seed=42)
    print(f"\nSampled {len(sampled)} documents:")
    for d in sampled[:3]:
        print(f"  [{d.source_dataset}] {d.id}: {d.document[:80]}...")
