"""Minimal working example: load -> chunk -> embed -> brute-force similarity.
    uv add openai python-dotenv

    uv run demo.py                         # MiniLM, calibration.json
    uv run demo.py --hash                  # offline HashEmbedder
    uv run demo.py --eval ../../data/eval/questions.json
    uv run demo.py -q "your question"      # ad-hoc questions instead of a file
    uv run demo.py --hash --min-score 0.29 # hash needs its own threshold
    uv run demo.py --generate              # + LLM answers; provider/model/key from .env
    uv run demo.py --generate --provider mistral
"""


from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ingestion import MAX_CHARS, load_documents, chunk_documents
from embedding import DEFAULT_MODEL, HashEmbedder, SentenceTransformerEmbedder, embed_chunks
from generation import PROVIDERS, GenerationResult, env, generate, make_llm
from retrieval import DEFAULT_K, DEFAULT_MIN_SCORE, Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]



def load_env_file(path: Path) -> None:
    """Load KEY=value pairs from .env; variables already set in the shell win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv not installed; using shell environment only "
              "(uv add python-dotenv)")
        return
    if path.exists():
        load_dotenv(path, override=False)
 
 
def load_cases(args: argparse.Namespace) -> list[dict]:
    if args.query:  # ad-hoc questions: nothing to check against
        return [{"id": f"adhoc{i}", "question": q} for i, q in enumerate(args.query, 1)]
    return json.loads(args.eval.read_text(encoding="utf-8"))
 
 
def check_generation(gen: GenerationResult, case: dict, stats: dict) -> None:
    """Print the generated answer and score it against the eval case."""
    if gen.abstained:
        print(f"  LLM: [abstained by {gen.abstained_by}]")
    else:
        print("  LLM: " + gen.answer.replace("\n", "\n       "))
 
    if "should_abstain" not in case:
        return
    stats["cases"] += 1
    ok = gen.abstained == bool(case["should_abstain"])
    stats["decisions_ok"] += ok
    notes = [] if ok else ["WRONG decision"]
 
    if not gen.abstained and not case["should_abstain"]:
        stats["answered"] += 1
        text = gen.answer.lower()
        missing = [t for t in case.get("required_terms", []) if t.lower() not in text]
        stats["terms_ok"] += not missing
        if missing:
            notes.append(f"missing terms {missing}")
        uncited = set(case.get("expected_sources", [])) - gen.cited_documents
        stats["sources_cited"] += not uncited
        if uncited:
            notes.append(f"expected source not cited {sorted(uncited)}")
    if not gen.abstained and not gen.citations:
        stats["uncited"] += 1
        notes.append("no citations")
    if gen.unknown_citations:
        stats["unknown_citations"] += len(gen.unknown_citations)
        notes.append(f"unknown citations {gen.unknown_citations}")
    print(f"  => {'ok' if not notes else '; '.join(notes)}")



def main() -> int:
    load_env_file(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=PROJECT_ROOT / "data" / "knowledge_base")
    parser.add_argument("--eval", type=Path,
                        default=PROJECT_ROOT / "data" / "evaluation" / "calibration.json")
    parser.add_argument("--hash", action="store_true", help="use the offline HashEmbedder")
    parser.add_argument("-q", "--query", action="append", help="repeatable; overrides --eval")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=None,
                        help="default: the calibrated value in retrieval.py")
    parser.add_argument("--store", default="qdrant", choices=["qdrant", "memory"],
                        help="qdrant: local Qdrant (VECTOR_STORAGE_PATH, empty = in memory); "
                             "memory: plain numpy")
    parser.add_argument("--generate", action="store_true", help="also generate LLM answers")
    parser.add_argument("--provider", default=None, choices=[*PROVIDERS, "anthropic"],
                        help="default: LLM_PROVIDER from .env")
    parser.add_argument("--model", default=None, help="default: <PROVIDER>_MODEL from .env")
    args = parser.parse_args()

    # 1. Load
    docs = load_documents(args.docs)
    print(f"Loaded {len(docs)} documents from {args.docs}")

    # 2. Chunk
    chunks = chunk_documents(docs)
    print(f"\n{len(chunks)} chunks (MAX_CHARS={MAX_CHARS}):")
    for c in chunks:
        print(f"  {c.chunk_id:<28} {len(c.text):>5} chars  {c.text[:50]!r}...")

    # 3. Embed
    embedder = HashEmbedder() if args.hash else SentenceTransformerEmbedder(
        model_name=env("EMBEDDING_MODEL", DEFAULT_MODEL),
        revision=env("EMBEDDING_REVISION"),
    )
    start = time.perf_counter()
    vectors = embed_chunks(chunks, embedder)
    elapsed = time.perf_counter() - start
    print(f"\nEmbedded with {embedder.model_name}: shape {vectors.shape}, "
          f"{vectors.dtype}, {elapsed:.2f}s")

    norms = np.linalg.norm(vectors, axis=1)
    checks = {
        "one vector per chunk": vectors.shape[0] == len(chunks),
        "no NaN/inf": bool(np.isfinite(vectors).all()),
        "unit length": bool(np.allclose(norms, 1.0, atol=1e-4)),
        "deterministic": bool(np.allclose(vectors, embed_chunks(chunks, embedder), atol=1e-5)),
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")

    # 4. Retrieve
    store = None
    if args.store == "qdrant":
        from qdrant_store import QdrantStore
 
        storage = env("VECTOR_STORAGE_PATH")
        path = (PROJECT_ROOT / storage) if storage else None  # relative to project root
        store = QdrantStore.open(chunks, embedder, path=path, vectors=vectors)
        print(f"\nVector store: Qdrant local, {path or 'in memory'}, "
              f"collection {store.collection} ({len(store)} points)")
    retriever = Retriever(chunks, embedder, k=args.k, min_score=args.min_score,
                          vectors=vectors)
    print(f"\nRetriever: k={retriever.k}, min_score={retriever.min_score}")
    llm = make_llm(args.provider, args.model) if args.generate else None
    if llm:
        print(f"Generator: {args.provider or env('LLM_PROVIDER', 'groq')} / {llm.model}")
    gen_stats = {"cases": 0, "decisions_ok": 0, "answered": 0, "terms_ok": 0,
                 "sources_cited": 0, "uncited": 0, "unknown_citations": 0}
 
    doc_ids = {c.document_id for c in chunks}
    stats = {"answerable": 0, "found": 0, "abstain_cases": 0, "decisions_ok": 0, "cases": 0}
    answer_scores: list[float] = []   # weakest expected source per answerable case
    abstain_scores: list[float] = []  # top score per abstain case
    overview: list[tuple[float, str, str, str]] = []  # (score, id, kind, question)
 
    for case in load_cases(args):
        result = retriever.retrieve(case["question"])
        candidates = retriever.search(case["question"])
        expected = set(case.get("expected_sources") or [])
        should_abstain = case.get("should_abstain")
 
        label = ("should abstain" if should_abstain
                 else "expected: " + ", ".join(sorted(expected)) if expected
                 else "no expectation")
        print(f"\n[{case['id']}] {case['question']}\n  ({label})")
        for hit in candidates:
            mark = "✓" if hit.chunk.document_id in expected else " "
            below = "" if result.hits else "   (question below min_score)"
            print(f"  {mark} {hit.score:.3f}  {hit.chunk.chunk_id}{below}")
 
        decision = "ABSTAIN" if result.abstain else "ANSWER"
        if "should_abstain" in case:
            stats["cases"] += 1
            ok = result.abstain == bool(should_abstain)
            stats["decisions_ok"] += ok
            print(f"  -> {decision} {'ok' if ok else 'WRONG'}")
        else:
            print(f"  -> {decision}")

        if llm:
            check_generation(generate(result, llm), case, gen_stats)
 
        if should_abstain:
            stats["abstain_cases"] += 1
            abstain_scores.append(result.top_score)
            overview.append((result.top_score, case["id"], "abstain", case["question"]))
        elif expected:
            stats["answerable"] += 1
            retrieved = {h.chunk.document_id for h in result.hits}
            stats["found"] += expected <= retrieved
            if expected - doc_ids:
                print(f"  ! expected source(s) not in knowledge base: {sorted(expected - doc_ids)}")
            else:
                best: dict[str, float] = {}  # best score per document
                for hit in retriever.search(case["question"], k=len(chunks)):
                    best.setdefault(hit.chunk.document_id, hit.score)
                answer_scores.append(min(best[d] for d in expected))
                overview.append((answer_scores[-1], case["id"], "answer", case["question"]))

    # Overview: every case by its deciding score, highest first.
    # answer  = score of its weakest expected source (must stay ABOVE min_score)
    # abstain = its top score (far off-topic cases must fall BELOW min_score)
    if overview:
        print(f"\nAll cases by deciding score (min_score={retriever.min_score}):")
        line_drawn = False
        for score, case_id, kind, question in sorted(overview, reverse=True):
            if not line_drawn and score < retriever.min_score:
                print(f"  {'-' * 20} min_score {retriever.min_score} {'-' * 20}")
                line_drawn = True
            print(f"  {score:.3f}  {kind:<7}  {case_id:<6}  {question[:60]}")
 
    # Summary
    if stats["cases"]:
        print(f"\nCorrect answer/abstain decisions: {stats['decisions_ok']}/{stats['cases']}")
    if stats["answerable"]:
        print(f"All expected sources retrieved (k={retriever.k}): "
              f"{stats['found']}/{stats['answerable']}")
    if answer_scores and abstain_scores:
        lowest, highest = min(answer_scores), max(abstain_scores)
        print(f"Lowest score of an expected source:   {lowest:.3f}")
        print(f"Highest top score of an abstain case: {highest:.3f}")
        if lowest > highest:
            print(f"-> any min_score between {highest:.3f} and {lowest:.3f} separates them")
        else:
            print("-> overlap: no single min_score separates answerable from abstain cases")

    if llm:
        g = gen_stats
        print(f"\nGeneration ({llm.model}):")
        print(f"  Correct final answer/abstain decisions: {g['decisions_ok']}/{g['cases']}")
        print(f"  Answers with all required terms:        {g['terms_ok']}/{g['answered']}")
        print(f"  Answers citing all expected sources:    {g['sources_cited']}/{g['answered']}")
        print(f"  Answers without any citation:           {g['uncited']}")
        print(f"  Citations of IDs that were not provided: {g['unknown_citations']}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())