"""Teacher LLM — generates QA pairs from knowledge passages.

Uses OpenRouter API with the paper's Prompt 2 format:
- Generates 3 QA pairs per passage
- Output: [{"question": ..., "answer": ..., "full_answer": ...}]

Supports pluggable backends:
- OpenRouter (default): any model via OpenRouter API
- Local: HuggingFace model (for offline use)
- Dummy: echoes passage (for testing)
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional


# Prompt from paper Appendix C.1 — Prompt 2: Question-Answer Generation
QA_GENERATION_PROMPT = """I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.

You need to generate the question and answer in the following format:
[
  {{
    "question": "What is the capital of France?",
    "answer": "Paris",
    "full_answer": "The capital of France is Paris."
  }},
]

This list should have at least three elements. You only need to output this list in the above format.

Passage: {passage}"""


class OpenRouterTeacher:
    """Teacher LLM via OpenRouter API."""

    def __init__(
        self,
        model: str = "openrouter/owl-alpha",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key=."
            )

    def generate_qa(self, passage: str) -> list[dict]:
        """Generate QA pairs from a passage.

        Returns:
            List of {"question": str, "answer": str, "full_answer": str}
        """
        prompt = QA_GENERATION_PROMPT.format(passage=passage[:3000])  # truncate long passages

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode()

        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"]
                return self._parse_qa_output(content)
        except Exception as e:
            print(f"  [Teacher] API error: {e}")
            return self._fallback_qa(passage)

    def _parse_qa_output(self, content: str) -> list[dict]:
        """Parse JSON output from the teacher LLM."""
        # Try to extract JSON array from the output
        content = content.strip()

        # Find JSON array in the output
        start = content.find("[")
        end = content.rfind("]")

        if start != -1 and end != -1 and end > start:
            try:
                qa_list = json.loads(content[start:end + 1])
                # Validate format
                valid = []
                for item in qa_list:
                    if "question" in item and "answer" in item:
                        valid.append({
                            "question": item["question"],
                            "answer": item.get("answer", ""),
                            "full_answer": item.get("full_answer", item.get("answer", "")),
                        })
                if valid:
                    return valid
            except json.JSONDecodeError:
                pass

        return self._fallback_qa(content)

    def _fallback_qa(self, passage: str) -> list[dict]:
        """Generate simple QA pairs when the LLM output can't be parsed."""
        sentences = [s.strip() for s in passage.split(".") if len(s.strip()) > 20]
        qa_pairs = []
        for i, sent in enumerate(sentences[:3]):
            qa_pairs.append({
                "question": f"What does the passage say about: {sent[:50]}...?",
                "answer": sent,
                "full_answer": sent,
            })
        return qa_pairs


class DummyTeacher:
    """Dummy teacher for testing — echoes the passage."""

    def generate_qa(self, passage: str) -> list[dict]:
        """Return passage as a single QA pair (for pipeline testing only)."""
        return [
            {
                "question": "What is this passage about?",
                "answer": passage[:500],
                "full_answer": passage[:500],
            }
        ]


def create_teacher(
    teacher_type: str = "openrouter",
    model: str = "openrouter/owl-alpha",
    api_key: str = "",
) -> OpenRouterTeacher | DummyTeacher:
    """Factory: create the right teacher based on config."""
    if teacher_type == "openrouter":
        return OpenRouterTeacher(model=model, api_key=api_key)
    elif teacher_type == "dummy":
        return DummyTeacher()
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")
