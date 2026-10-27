
import csv
from pathlib import Path

import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

CHUNK_SIZE = 40
OVERLAP = 10


def chunk_documents(overlap=OVERLAP):
    """Split policy documents into fixed-size word chunks."""
    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError("Overlap must be between 0 and 39.")

    chunks = []
    step = CHUNK_SIZE - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunks.append({
                "source": path.name,
                "text": " ".join(words[start:end]),
            })

            if end >= len(words):
                break

            start += step

    return chunks


class PolicyRAG:
    def __init__(self, overlap=OVERLAP):
        self.chunks = chunk_documents(overlap=overlap)

        if not self.chunks:
            raise ValueError(f"No policy documents found in {POLICY_DIR}")

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        """Return the k highest-scoring policy chunks."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        # Stable ordering makes ties deterministic.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i)
        )

        results = []
        for index in ranked_indices[:min(k, len(ranked_indices))]:
            results.append({
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            })

        return results

    def answer(self, question):
        """Answer using only retrieved policy context."""
        results = self.retrieve(question, k=3)

        context_chunks = "\n".join(
            f"[{item['source']}]\n{item['text']}"
            for item in results
        )

        user_content = (
            f"CONTEXT:\n{context_chunks}\n"
            f"QUESTION:\n{question}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT: Answer only from the supplied "
                    "policy context. If the context does not contain "
                    "the answer, reply beginning exactly with "
                    "'I could not find'. Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        response = mock_llm.chat(messages)
        text = response.choices[0].message.content

        return {
            "answer": text,
            "sources": list(dict.fromkeys(
                item["source"] for item in results
            )),
        }


def evaluate(rag):
    """Measure retrieval, expected facts, and unanswerable fallbacks."""
    with EVAL_FILE.open(encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))

    answerable = [row for row in questions if row["expected_doc"] != "NONE"]
    unanswerable = [row for row in questions if row["expected_doc"] == "NONE"]

    retrieval_hits = 0
    fact_hits = 0
    fallback_hits = 0
    failures = []

    for row in questions:
        question = row["question"]
        results = rag.retrieve(question, k=3)
        result = rag.answer(question)

        if row["expected_doc"] == "NONE":
            if result["answer"].startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append(
                    (question, "Expected an unanswerable fallback", result["answer"])
                )
            continue

        if row["expected_doc"] in [item["source"] for item in results]:
            retrieval_hits += 1
        else:
            failures.append(
                (question, "Expected document missing from top three",
                 [item["source"] for item in results])
            )

        # Required by the assignment: case-sensitive substring check.
        if row["expected_fact"] in result["answer"]:
            fact_hits += 1
        else:
            failures.append(
                (question, f"Expected fact: {row['expected_fact']}",
                 result["answer"])
            )

    return {
        "total_chunks": len(rag.chunks),
        "answerable": len(answerable),
        "retrieval_hits": retrieval_hits,
        "fact_hits": fact_hits,
        "unanswerable": len(unanswerable),
        "fallback_hits": fallback_hits,
        "failures": failures,
    }


def print_evaluation(label, rag):
    result = evaluate(rag)

    print(f"\n=== {label} ===")
    print(f"Total chunks: {result['total_chunks']}")
    print(
        f"Answerable questions with expected document in top 3: "
        f"{result['retrieval_hits']}/{result['answerable']}"
    )
    print(
        f"Answers containing expected fact: "
        f"{result['fact_hits']}/{result['answerable']}"
    )
    print(
        f"Unanswerable questions with correct fallback: "
        f"{result['fallback_hits']}/{result['unanswerable']}"
    )

    if result["failures"]:
        print("\nFailures:")
        for question, reason, actual in result["failures"]:
            print(f"- Question: {question}")
            print(f"  Reason: {reason}")
            print(f"  Actual: {actual}")

    return result


if __name__ == "__main__":
    # Task 1: chunk count and source distribution.
    chunks = chunk_documents(overlap=10)
    print("=== TASK 1: CHUNK COUNTS ===")
    print(f"Total chunks: {len(chunks)}")

    counts = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    for source, count in sorted(counts.items()):
        print(f"{source}: {count}")

    # Task 2: inspect the reference retrieval query.
    rag = PolicyRAG(overlap=10)
    print("\n=== TASK 2: REFERENCE RETRIEVAL ===")
    for item in rag.retrieve("refund policy late delivery", k=3):
        print(f"{item['source']} | score={item['score']:.4f}")
        print(item["text"])

    # Tasks 3–4: answer and evaluate using overlapping chunks.
    mock_llm.reset()
    print_evaluation("OVERLAP = 10", rag)

    # Task 4: compare with non-overlapping chunks.
    no_overlap_rag = PolicyRAG(overlap=0)
    mock_llm.reset()
    print_evaluation("OVERLAP = 0", no_overlap_rag)

    print(
        "\nTF-IDF explanation: Term Frequency (TF) measures how often "
        "a term occurs in a chunk, while Inverse Document Frequency "
        "(IDF) gives less weight to terms common across many chunks. "
        "A limitation is that TF-IDF relies on word overlap, so two "
        "phrases with the same meaning but different vocabulary may "
        "not retrieve each other well."
    )
