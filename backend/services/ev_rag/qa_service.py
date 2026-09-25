"""
qa_service.py — EV RAG wrapper with text cleaning.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List

try:
    from .retriever import EVRetriever, load_chunks
    from .answerer import get_answer
    from .text_cleaner import clean_answer
except ImportError:
    from retriever import EVRetriever, load_chunks
    from answerer import get_answer
    from text_cleaner import clean_answer


class EVRAGService:
    HIGH_CONFIDENCE = 15.0
    MIN_CONFIDENCE = 3.0

    def __init__(self):
        self.data_dir = Path(__file__).parent / "data" / "entities"
        self.retrievers: List[Dict] = []
        self.loaded = False
        self._load_all_manuals()

    # -----------------------------------------------------------------
    def _load_all_manuals(self):
        if not self.data_dir.exists():
            print(f"⚠️ EV_RAG data folder not found: {self.data_dir}")
            return

        json_files = sorted(self.data_dir.glob("*_chunks.json"))
        if not json_files:
            print(f"⚠️ No chunk files found in: {self.data_dir}")
            return

        print(f"\n📚 Loading EV manuals ({len(json_files)} file(s))...")

        for json_path in json_files:
            try:
                chunks = load_chunks(str(json_path))
                if not chunks:
                    continue

                retriever = EVRetriever(chunks)
                manual_name = json_path.stem.replace("_chunks", "")
                self.retrievers.append({
                    "name": manual_name,
                    "retriever": retriever,
                })
                print(f"   ✅ {manual_name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"   ❌ {json_path.name}: {e}")

        self.loaded = len(self.retrievers) > 0
        if self.loaded:
            print(f"✅ EV_RAG ready with {len(self.retrievers)} manual(s)\n")

    # -----------------------------------------------------------------
    def ask(self, question: str, top_k: int = 10) -> Optional[Dict]:
        if not self.loaded:
            return None

        best_overall = None
        best_score = 0.0

        for entry in self.retrievers:
            manual_name = entry["name"]
            retriever = entry["retriever"]

            try:
                results = retriever.search(question, top_k=top_k)
            except Exception:
                continue

            if not results:
                continue

            answer = get_answer(question, results)
            if not answer:
                continue

            chunk = answer["chunk"]
            score = answer["sentence_score"] + answer["retrieval_score"]

            if score > best_score:
                best_score = score
                # Clean the raw manual text
                cleaned = clean_answer(answer["answer"])

                best_overall = {
                    "answer": cleaned,
                    "raw_answer": answer["answer"],
                    "manual": (
                        chunk.get("document")
                        or chunk.get("filename")
                        or chunk.get("file_name")
                        or chunk.get("source")
                        or manual_name
                    ),
                    "page_start": chunk.get("page_start") or chunk.get("page"),
                    "page_end": chunk.get("page_end"),
                    "score": round(score, 3),
                }

        if best_overall:
            if best_score >= self.HIGH_CONFIDENCE:
                best_overall["confidence"] = "high"
            elif best_score >= self.MIN_CONFIDENCE:
                best_overall["confidence"] = "medium"
            else:
                best_overall["confidence"] = "low"

        return best_overall

    # -----------------------------------------------------------------
    @staticmethod
    def format_answer(result: Dict, include_source: bool = True) -> str:
        if not result:
            return "I couldn't find that in the EV manuals."

        text = f"🚗 {result['answer']}"

        if include_source:
            source = result.get("manual", "")
            page_start = result.get("page_start")
            page_end = result.get("page_end")

            if source:
                text += f"\n\n📖 Source: {source}"
                if page_start and page_end:
                    text += f" (pages {page_start}–{page_end})"
                elif page_start:
                    text += f" (page {page_start})"

        return text

    def list_manuals(self) -> List[str]:
        return [entry["name"] for entry in self.retrievers]


if __name__ == "__main__":
    ev = EVRAGService()
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() == "exit":
            break
        result = ev.ask(q)
        print(EVRAGService.format_answer(result))