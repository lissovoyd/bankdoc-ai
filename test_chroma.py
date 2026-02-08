import chromadb
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import os
import torch

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

torch.set_num_threads(2)
torch.set_num_interop_threads(1)



CHROMA_DB_DIR = Path("chroma_db")
MODELS_CACHE_DIR = Path("models_cache")

# Use the SAME cached local model as tasks.py (no re-download if already cached)
embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    cache_folder=str(MODELS_CACHE_DIR)
)

def label_from_distance(d: float) -> str:
    # tuned for normalized embeddings + L2 (unit vectors)
    if d <= 0.60:
        return "🎯 Bullseye"
    if d <= 0.90:
        return "✅ Highly relevant"
    if d <= 1.20:
        return "🟡 Somewhat relevant"
    return "❌ Ignore"

def extract_context(text: str, query: str, context_words: int = 8):
    words = re.findall(r"\S+", text)
    words_lower = [w.lower() for w in words]
    query_lower = query.lower()

    contexts = []
    for i, word in enumerate(words_lower):
        if query_lower in word:
            start = max(0, i - context_words)
            end = min(len(words), i + context_words + 1)

            snippet = []
            for j in range(start, end):
                snippet.append(f">>>{words[j]}<<<" if j == i else words[j])
            contexts.append(" ".join(snippet))
    return contexts

def run_query(collection, query: str, top_k: int = 5, cutoff: float = 1.2):
    # embed query with normalization (must match how you embedded chunks)
    query_emb = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    print("\n\n")
    print("\n" + "=" * 120)
    print(f"🔍 QUERY: {query}")
    print("=" * 120)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    any_printed = False

    for rank, (chunk_text, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        if dist > cutoff:
            continue  # disregard anything 1.2+

        any_printed = True
        label = label_from_distance(dist)

        print("-" * 120)
        print(f"Rank #{rank} | {label} | DISTANCE: {dist:.4f}")
        print(f"Document: {meta.get('title')} | Page {meta.get('page_num')} | Chunk #{meta.get('chunk_index')}")
        print("-" * 120)

        # Try to highlight direct matches (works best for short keyword queries)
        contexts = extract_context(chunk_text, query)

        if contexts:
            print("Direct keyword hit in chunk:")
            for i, ctx in enumerate(contexts[:3], start=1):  # limit spam
                print(f"  [{i}] ...{ctx}...")
        else:
            print("Preview:")
            print(chunk_text)

    if not any_printed:
        best = dists[0] if dists else None
        print(f"⚠ No results under cutoff {cutoff}. Best distance was: {best:.4f}" if best is not None else
              f"⚠ No results under cutoff {cutoff}.")

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("bankdoc")

    count = collection.count()
    print(f"✓ ChromaDB has {count} chunks stored")
    if count == 0:
        raise SystemExit

    # 4–5 built-in test cases (mix of “should match” and “should not match”)
    TEST_QUERIES = [
        # Ending / “body too heavy” (should match strongly)
        "он сказал: мое тело слишком тяжелое, мне его не унести. это не смерть, а возвращение домой к своей звезде",
        # Snake scene keyword (should find direct hit if that chunk exists)
        "змея в пустыне, опасность, укус, страх, смерть",
        # Water / well scene (should match if you have the well passage)
        "колодец вода ведро напиться",
        # Roses theme (may or may not be present depending on your excerpt)
        "роза стеклянный колпак барашек",
        # Irrelevant (should be ignored by cutoff)
        "как приготовить борщ и какие ингредиенты нужны",
    ]

    for q in TEST_QUERIES:
        run_query(collection, q, top_k=3, cutoff=1.25)
