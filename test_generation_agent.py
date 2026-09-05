"""
Test Generation Agent
======================

An agent that:
  1. Stores labelled Q&A examples in a Chroma vector store, with metadata
     (concept, question, answer, user_answer, feedback, timestamp).
     feedback is one of: "acceptable", "too difficult", "too easy", "not related".
  2. Given a concept + a few example Q&A pairs, uses a Gemma chat model
     (via langchain_google_genai.ChatGoogleGenerativeAI) to generate a new
     question/answer pair for that concept.
  3. Quality-gates every new example: it embeds the new question, retrieves
     the 5 most similar stored examples, and only accepts the new example if
     the majority of those neighbours were labelled "acceptable". Otherwise
     it regenerates (up to max_retries).
  4. Produces a batched summary (groups of 10, sorted by timestamp) counting
     acceptable / too difficult / too easy / not related examples, plus an
     optional stacked bar chart.

Install:
    pip install langchain-google-genai langchain-chroma langchain-core chromadb matplotlib

Auth:
    Set GOOGLE_API_KEY in the environment, or pass google_api_key=... explicitly.

Note on the model name:
    "gemma-4" is used below as a placeholder. Swap `model_name` for whatever
    Gemma model id you actually have access to in Google's Generative AI API
    (naming has changed across Gemma releases, e.g. "gemma-3-27b-it").
"""

import os
import json
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import SentenceTransformerEmbeddings


VALID_FEEDBACK = {"acceptable", "too difficult", "too easy", "not related"}


class TestGenerationAgent:
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        model_name: str = "gemma-4-26b-a4b-it",  # <-- swap for the exact Gemma model id you have access to
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", # "models/embedding-001",
        persist_directory: str = "./chroma_store",
        collection_name: str = "test_examples",
        temperature: float = 0.7,
        similarity_k: int = 5,
        max_retries: int = 5,
    ):
        api_key = google_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Provide GEMINI_API_KEY or set the GEMINI_API_KEY env var.")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model=embedding_model,
        #     google_api_key=api_key,
        # )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        self.similarity_k = similarity_k
        self.max_retries = max_retries

    # ---------------------------------------------------------------- #
    # Generation
    # ---------------------------------------------------------------- #

    def _build_prompt(self, concept: str, examples: List[Dict[str, str]]) -> List:
        example_block = "\n\n".join(
            f"Example {i + 1}:\nQuestion: {ex['question']}\nAnswer: {ex['answer']}"
            for i, ex in enumerate(examples)
        )
        system = SystemMessage(content=(
            "You are a test-question generator. Given a concept and a few example "
            "question/answer pairs, produce ONE new question and its correct answer "
            "that tests the same concept. The new question must not duplicate any "
            "example. Respond with STRICT JSON only, no markdown fences, in the form:\n"
            '{"question": "...", "answer": "..."}'
        ))
        human = HumanMessage(content=(
            f"Concept: {concept}\n\n"
            f"Existing examples:\n{example_block}\n\n"
            "Generate a new question and answer for this concept as JSON."
        ))
        return [system, human]

    def _call_llm_for_example(self, concept: str, examples: List[Dict[str, str]]) -> Dict[str, str]:
        messages = self._build_prompt(concept, examples)
        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        texts.append(part["text"])
                    elif "text" in part:
                        texts.append(part["text"])
                    elif "content" in part:
                        texts.append(part["content"])
                elif hasattr(part, "text"):
                    texts.append(getattr(part, "text"))
            text = "\n".join(texts)
        else:
            text = str(content)

        text = text.strip()

        # Strip accidental markdown code fences if present.
        if "```" in text:
            import re
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()
            else:
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx : end_idx + 1]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    raise ValueError(f"Model did not return valid JSON: {text!r}") from e
            else:
                raise ValueError(f"Model did not return valid JSON: {text!r}") from e

        if "question" not in data or "answer" not in data:
            raise ValueError(f"Model JSON missing required keys: {data}")

        return {"question": str(data["question"]).strip(), "answer": str(data["answer"]).strip()}


    def find_similar(
        self,
        question: str,
        concept: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        k = k or self.similarity_k
        query = f"Concept: {concept}\nQuestion: {question}" if concept else question
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception:
            return []

    @staticmethod
    def _majority_is_acceptable(docs: List[Document]) -> bool:
        if not docs:
            # Cold start: nothing to compare against yet, so accept.
            return True
        counts = defaultdict(int)
        for d in docs:
            fb = d.metadata.get("feedback")
            if fb in VALID_FEEDBACK:
                counts[fb] += 1
        if not counts:
            # None of the neighbours have feedback yet -> accept.
            return True
        top_feedback, top_count = max(counts.items(), key=lambda kv: kv[1])
        return top_feedback == "acceptable" and top_count > (len(docs) / 2)

    def generate_example(
        self,
        concept: str,
        few_shot_examples: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Generate a new (question, answer) pair for `concept`.

        Regenerates until the 5 nearest stored neighbours (by question
        similarity) are majority-"acceptable", or until max_retries is hit
        (in which case the last candidate is stored anyway, flagged
        accepted_on_similarity_check=False, for manual review).
        """
        last_candidate, last_similar = None, []

        for attempt in range(1, self.max_retries + 1):
            candidate = self._call_llm_for_example(concept, few_shot_examples)
            similar_docs = self.find_similar(candidate["question"], concept=concept)
            last_candidate, last_similar = candidate, similar_docs

            if self._majority_is_acceptable(similar_docs):
                doc_id = self._store_example(
                    concept=concept,
                    question=candidate["question"],
                    answer=candidate["answer"],
                )
                return {
                    "id": doc_id,
                    "concept": concept,
                    "question": candidate["question"],
                    "answer": candidate["answer"],
                    "attempts": attempt,
                    "accepted_on_similarity_check": True,
                    "similar_examples": [d.metadata for d in similar_docs],
                }

        # Retries exhausted: store anyway but flag it for a human to look at.
        doc_id = self._store_example(
            concept=concept,
            question=last_candidate["question"],
            answer=last_candidate["answer"],
        )
        return {
            "id": doc_id,
            "concept": concept,
            "question": last_candidate["question"],
            "answer": last_candidate["answer"],
            "attempts": self.max_retries,
            "accepted_on_similarity_check": False,
            "similar_examples": [d.metadata for d in last_similar],
        }

    # ---------------------------------------------------------------- #
    # Storage
    # ---------------------------------------------------------------- #

    def _store_example(
        self,
        concept: str,
        question: str,
        answer: str,
        user_answer: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        if feedback is not None and feedback not in VALID_FEEDBACK:
            raise ValueError(f"feedback must be one of {VALID_FEEDBACK}, got {feedback!r}")

        doc_id = str(uuid.uuid4())
        metadata = {
            "concept": concept,
            "question": question,
            "answer": answer,
            "user_answer": user_answer or "",
            "feedback": feedback or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        page_content = f"Concept: {concept}\nQuestion: {question}" if concept else question
        document = Document(page_content=page_content, metadata=metadata)
        self.vectorstore.add_documents([document], ids=[doc_id])
        return doc_id

    def add_manual_example(
        self,
        concept: str,
        question: str,
        answer: str,
        user_answer: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """Seed the store directly, e.g. with human-authored/labelled examples."""
        return self._store_example(concept, question, answer, user_answer, feedback)

    def record_feedback(
        self,
        doc_id: str,
        feedback: str,
        user_answer: Optional[str] = None,
    ) -> None:
        """Attach human feedback (and optionally the user's own answer) to a stored example."""
        if feedback not in VALID_FEEDBACK:
            raise ValueError(f"feedback must be one of {VALID_FEEDBACK}, got {feedback!r}")

        existing = self.vectorstore.get(ids=[doc_id], include=["metadatas", "documents"])
        if not existing["ids"]:
            raise KeyError(f"No stored example with id {doc_id}")

        metadata = existing["metadatas"][0]
        metadata["feedback"] = feedback
        if user_answer is not None:
            metadata["user_answer"] = user_answer

        # Chroma has no in-place metadata update via this API -> delete + re-add same id.
        page_content = existing["documents"][0]
        self.vectorstore.delete(ids=[doc_id])
        self.vectorstore.add_documents(
            [Document(page_content=page_content, metadata=metadata)], ids=[doc_id]
        )

    # ---------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------- #

    def get_all_records(self) -> List[Dict[str, Any]]:
        raw = self.vectorstore.get(include=["metadatas"])
        records = raw["metadatas"]
        records.sort(key=lambda m: m.get("timestamp", ""))
        return records

    def summarize_in_batches(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Sort all stored records by timestamp, chunk into groups of `batch_size`,
        and count feedback categories per group.
        """
        records = self.get_all_records()
        summary = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            counts = {
                "acceptable": 0,
                "too difficult": 0,
                "too easy": 0,
                "not related": 0,
                "unlabeled": 0,
            }
            for r in batch:
                fb = r.get("feedback", "")
                if fb in counts:
                    counts[fb] += 1
                else:
                    counts["unlabeled"] += 1
            summary.append({
                "batch_index": i // batch_size,
                "start_timestamp": batch[0].get("timestamp"),
                "end_timestamp": batch[-1].get("timestamp"),
                "count": len(batch),
                **counts,
            })
        return summary

    def plot_summary(self, batch_size: int = 10, save_path: Optional[str] = None):
        """Stacked bar chart of feedback counts per batch. Requires matplotlib."""
        import matplotlib.pyplot as plt

        summary = self.summarize_in_batches(batch_size)
        if not summary:
            print("No records to visualize.")
            return summary

        labels = [f"Batch {s['batch_index']}" for s in summary]
        categories = ["acceptable", "too difficult", "too easy", "not related", "unlabeled"]
        bottoms = [0] * len(summary)

        fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.8), 5))
        for cat in categories:
            values = [s[cat] for s in summary]
            ax.bar(labels, values, bottom=bottoms, label=cat)
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_ylabel("Number of examples")
        ax.set_title(f"Feedback distribution per batch of {batch_size}")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        return summary


# ---------------------------------------------------------------------- #
# Demo
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    agent = TestGenerationAgent(
        model_name="gemma-4-26b-a4b-it",              # replace with an actual available Gemma model id
        persist_directory="./chroma_store",
        collection_name="test_examples",
    )

    # Seed a few labelled examples so the similarity gate has something to compare against.
    agent.add_manual_example(
        concept="Python list comprehensions",
        question="What does [x**2 for x in range(5)] evaluate to?",
        answer="[0, 1, 4, 9, 16]",
        feedback="acceptable",
    )
    agent.add_manual_example(
        concept="Python list comprehensions",
        question="Write a comprehension that squares even numbers from 0 to 10.",
        answer="[x**2 for x in range(11) if x % 2 == 0]",
        feedback="acceptable",
    )

    few_shot = [
        {
            "question": "What does [x**2 for x in range(5)] evaluate to?",
            "answer": "[0, 1, 4, 9, 16]",
        },
        {
            "question": "Write a comprehension that squares even numbers from 0 to 10.",
            "answer": "[x**2 for x in range(11) if x % 2 == 0]",
        },
    ]

    result = agent.generate_example("Python list comprehensions", few_shot)
    print("\nnew example:\n"+"*"*20)
    print(json.dumps(result, indent=2))
    print("*"*20+"\n")

    # Simulate labelling the new example, then view the batched summary.
    agent.record_feedback(result["id"], feedback="acceptable")

    summary = agent.summarize_in_batches(batch_size=10)
    print("\nsummary:\n"+"*"*20)
    print(json.dumps(summary, indent=2))
    print("*"*20+"\n")

    # agent.plot_summary(batch_size=10, save_path="feedback_summary.png")
