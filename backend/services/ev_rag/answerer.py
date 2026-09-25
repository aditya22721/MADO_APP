"""
answerer.py

Extractive answer selection for EV manuals.

The answer is taken directly from the manual.
No LLM or generative AI is used.
"""

import re


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# SENTENCE SCORE
# ============================================================

def sentence_score(sentence, question):

    question_tokens = set(
        tokenize(question)
    )

    sentence_tokens = set(
        tokenize(sentence)
    )

    if not question_tokens or not sentence_tokens:
        return 0

    overlap = (
        question_tokens &
        sentence_tokens
    )

    return len(overlap)


# ============================================================
# ANSWER FROM CHUNK
# ============================================================

def extract_answer(question, chunk):

    text = chunk.get("text", "")

    sentences = split_sentences(text)

    if not sentences:
        return None

    scored_sentences = []

    for sentence in sentences:

        score = sentence_score(
            sentence,
            question
        )

        scored_sentences.append(
            (score, sentence)
        )

    scored_sentences.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_sentence = (
        scored_sentences[0]
    )

    # No meaningful word overlap
    if best_score == 0:
        return None

    return {
        "answer": best_sentence,
        "score": best_score
    }


# ============================================================
# ANSWER FROM RETRIEVED RESULTS
# ============================================================

def get_answer(question, retrieved_results):

    candidates = []

    for result in retrieved_results:

        chunk = result["chunk"]

        answer = extract_answer(
            question,
            chunk
        )

        if answer:

            candidates.append({

                "answer": answer["answer"],

                "sentence_score":
                    answer["score"],

                "retrieval_score":
                    result["final_score"],

                "chunk": chunk
            })

    if not candidates:
        return None

    # Highest sentence relevance first
    candidates.sort(
        key=lambda x: (
            x["sentence_score"],
            x["retrieval_score"]
        ),
        reverse=True
    )

    return candidates[0]