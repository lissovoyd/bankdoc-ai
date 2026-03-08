"""Offline RAG evaluation script.

Usage:
    python eval/evaluate.py [--dataset eval/golden_dataset.json] [--output eval/results.json]

Requires a running BankDoc AI server (uvicorn main:app) with documents already embedded.
Measures: faithfulness, context relevance, citation accuracy, decline accuracy.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


API_BASE = "http://localhost:8000"


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    if not items:
        print("ERROR: No items in golden dataset.")
        sys.exit(1)
    return items


def get_doc_id_by_filename(filename: str) -> int | None:
    """Look up doc_id from the API by filename."""
    resp = requests.get(f"{API_BASE}/api/docs")
    resp.raise_for_status()
    for doc in resp.json():
        if doc["filename"] == filename:
            return doc["id"]
    return None


def ask_question(doc_id: int, question: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/api/docs/{doc_id}/ask",
        json={"question": question},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def evaluate_item(item: dict, result: dict) -> dict:
    """Score a single Q&A pair against the golden answer."""
    scores = {}

    # Citation accuracy: do cited source pages match expected pages?
    actual_pages = {s["page_num"] for s in result.get("sources", [])}
    expected_pages = set(item.get("expected_pages", []))
    if expected_pages:
        overlap = actual_pages & expected_pages
        scores["citation_accuracy"] = len(overlap) / len(expected_pages) if expected_pages else 0.0
    else:
        scores["citation_accuracy"] = None  # no expected pages to compare

    # Decline accuracy: for "not_in_doc" items, did the system decline?
    if item.get("category") == "not_in_doc":
        scores["decline_correct"] = result.get("declined", False)
    else:
        scores["decline_correct"] = None

    # Context relevance: max re-ranker score (higher = more relevant context retrieved)
    scores["context_relevance"] = result.get("confidence", 0.0)

    # Faithfulness: self-assessed by the LLM (if structured output was used)
    # This is a proxy; for true faithfulness use RAGAS when available
    scores["grounded"] = result.get("grounded", True)

    return scores


def run_evaluation(dataset_path: str, output_path: str):
    items = load_dataset(dataset_path)
    print(f"Loaded {len(items)} evaluation items from {dataset_path}")

    results = []
    totals = {"citation_accuracy": [], "context_relevance": [], "decline_correct": []}

    for item in items:
        doc_id = get_doc_id_by_filename(item["document"])
        if doc_id is None:
            print(f"  SKIP: document '{item['document']}' not found in API")
            continue

        print(f"  [{item['id']}] {item['question'][:60]}...", end=" ")
        t0 = time.time()

        try:
            result = ask_question(doc_id, item["question"])
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        elapsed = time.time() - t0
        scores = evaluate_item(item, result)

        entry = {
            "id": item["id"],
            "question": item["question"],
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "answer": result.get("answer", "")[:200],
            "confidence": result.get("confidence", 0.0),
            "declined": result.get("declined", False),
            "elapsed_sec": round(elapsed, 2),
            "scores": scores,
        }
        results.append(entry)
        print(f"OK ({elapsed:.1f}s, conf={result.get('confidence', 0):.2f})")

        if scores["citation_accuracy"] is not None:
            totals["citation_accuracy"].append(scores["citation_accuracy"])
        totals["context_relevance"].append(scores["context_relevance"])
        if scores["decline_correct"] is not None:
            totals["decline_correct"].append(1.0 if scores["decline_correct"] else 0.0)

    # Aggregate
    summary = {}
    for key, values in totals.items():
        summary[key] = round(sum(values) / len(values), 4) if values else 0.0

    report = {
        "dataset": dataset_path,
        "total_items": len(items),
        "evaluated": len(results),
        "summary": summary,
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nResults saved to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BankDoc AI RAG Evaluation")
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    run_evaluation(args.dataset, args.output)
