"""Build the abstention dataset: retrieval features for every golden question.

Label: 1 if the question should be answered (class in_topic_answerable), else 0
(in-topic but unanswerable, off-topic, adversarial). Features come from the
vector search only (app/services/features.py), with the same in-force filter as
the retriever. The `split` column keeps the held-out questions apart: training
uses the visible split only.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma

from app.config import get_settings
from app.services.features import FEATURE_K, FEATURE_NAMES, extract_features
from app.services.ingestion import get_embedding

OUTPUT = ROOT / "eval" / "abstention_dataset.csv"
SPLITS = {"visible": "golden.jsonl", "held_out": "golden_held_out.jsonl"}


def label_for(record_class: str) -> int:
    return int(record_class == "in_topic_answerable")


def build_rows(store, search_filter) -> list[dict]:
    rows = []
    for split, name in SPLITS.items():
        with open(ROOT / "eval" / name) as f:
            records = [json.loads(line) for line in f if line.strip()]
        for record in records:
            results = store.similarity_search_with_score(
                record["question"], k=FEATURE_K, filter=search_filter
            )
            features = extract_features(
                [score for _, score in results], record["question"]
            )
            rows.append(
                {
                    "id": record["id"],
                    "split": split,
                    "class": record["class"],
                    "label": label_for(record["class"]),
                    **features,
                }
            )
    return rows


def main() -> None:
    settings = get_settings()
    store = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=get_embedding(settings),
    )
    search_filter = None if settings.include_superseded else {"in_force": True}
    rows = build_rows(store, search_filter)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "split", "class", "label", *FEATURE_NAMES]
        )
        writer.writeheader()
        writer.writerows(rows)

    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        positives = sum(r["label"] for r in part)
        print(
            f"{split}: {len(part)} rows, {positives} to answer, {len(part) - positives} to refuse"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
