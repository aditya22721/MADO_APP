"""
retriever.py

Retrieves relevant chunks from EV manuals using:
1. BM25
2. Keyword matching
3. Extracted entity matching

No LLM is used.
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(json_path):
    """
    Load chunks from a JSON file.

    Supports either:
        [ {...}, {...} ]

    or:
        {"chunks": [ {...}, {...} ]}
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data

    elif isinstance(data, dict):
        chunks = data.get("chunks", [])

    else:
        raise ValueError("Unsupported JSON structure.")

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase tokens.
    """

    if not text:
        return []

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# RETRIEVER
# ============================================================

class EVRetriever:

    def __init__(self, chunks):

        self.chunks = chunks

        # ----------------------------------------------------
        # Create BM25 corpus
        # ----------------------------------------------------

        self.corpus = []

        for chunk in chunks:

            text = chunk.get("text", "")
            heading = chunk.get("heading", "")

            combined_text = f"{heading} {text}"

            self.corpus.append(
                tokenize(combined_text)
            )

        # Avoid completely empty documents
        self.corpus = [
            tokens if tokens else ["empty"]
            for tokens in self.corpus
        ]

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(self.corpus)

        print("BM25 index ready.")

    # ========================================================
    # ENTITY TEXT
    # ========================================================

    def get_entity_texts(self, chunk):

        entities = chunk.get("entities", [])

        result = []

        for entity in entities:

            if isinstance(entity, dict):

                text = entity.get("text", "")

                if text:
                    result.append(text.lower())

            elif isinstance(entity, str):

                result.append(entity.lower())

        return result

    # ========================================================
    # KEYWORD MATCH
    # ========================================================

    def keyword_score(self, query_tokens, chunk):

        text = chunk.get("text", "").lower()

        heading = chunk.get("heading", "").lower()

        combined = f"{heading} {text}"

        score = 0

        for token in query_tokens:

            if token in combined:
                score += 1

        return score

    # ========================================================
    # ENTITY MATCH
    # ========================================================

    def entity_score(self, query_tokens, chunk):

        entities = self.get_entity_texts(chunk)

        if not entities:
            return 0

        score = 0

        for entity in entities:

            entity_tokens = tokenize(entity)

            overlap = set(query_tokens) & set(entity_tokens)

            score += len(overlap)

        return score

    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, question, top_k=5):

        query_tokens = tokenize(question)

        if not query_tokens:
            return []

        # ----------------------------------------------------
        # BM25 scores
        # ----------------------------------------------------

        bm25_scores = self.bm25.get_scores(query_tokens)

        results = []

        # ----------------------------------------------------
        # Combine scores
        # ----------------------------------------------------

        for index, chunk in enumerate(self.chunks):

            bm25_score = float(
                bm25_scores[index]
            )

            keyword_score = self.keyword_score(
                query_tokens,
                chunk
            )

            entity_score = self.entity_score(
                query_tokens,
                chunk
            )

            # Weighted score
            final_score = (
                bm25_score
                + (keyword_score * 1.5)
                + (entity_score * 2.0)
            )

            results.append({

                "chunk": chunk,

                "bm25_score": round(
                    bm25_score,
                    4
                ),

                "keyword_score": keyword_score,

                "entity_score": entity_score,

                "final_score": round(
                    final_score,
                    4
                )
            })

        # ----------------------------------------------------
        # Sort by relevance
        # ----------------------------------------------------

        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results[:top_k]


# ============================================================
# BUILD RETRIEVER
# ============================================================

def create_retriever(json_path):

    chunks = load_chunks(json_path)

    return EVRetriever(chunks)