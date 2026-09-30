# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for :func:`cola_dlm.apply_prompt_template`.

Social IQa (SIQA) examples are under-determined without their situational
context: e.g. "What does Tracy need to do before this?" only becomes answerable
once "Tracy didn't go home that evening and resisted Riley's attacks." is in the
prompt. These tests guard against silently serializing SIQA (and RACE) prompts
that drop the passage/context before scoring or exporting.
"""

from __future__ import annotations

import pytest

from cola_dlm import apply_prompt_template


def _render(task: str, item: dict) -> str:
    return apply_prompt_template(
        task=task,
        context=item.get("context", ""),
        question=item.get("question", ""),
        answer=item.get("ground_truth", item.get("answer", "")),
        choices=item.get("choices"),
    )


# A real SIQA example (matching the exported eval row id=0) plus its original
# Social IQa context, which the benchmark must preserve.
SIQA_ITEM = {
    "id": 0,
    "context": "Tracy didn't go home that evening and resisted Riley's attacks.",
    "question": "What does Tracy need to do before this?",
    "choices": ["make a new plan", "Go home and see Riley", "Find somewhere to go"],
    "ground_truth": "Find somewhere to go",
}


def test_siqa_prompt_includes_context() -> None:
    """The rendered SIQA prompt must carry the situational context.

    A prompt that only contains the question and choices is under-determined,
    so we require both a "Context:" label and the verbatim context text.
    """
    prompt = _render("siqa", SIQA_ITEM)

    assert "Context:" in prompt, f"SIQA prompt is missing a 'Context:' label:\n{prompt}"
    assert SIQA_ITEM["context"] in prompt, f"SIQA prompt dropped the original context text:\n{prompt}"


def test_siqa_context_precedes_question() -> None:
    """Context must appear before the question it disambiguates."""
    prompt = _render("siqa", SIQA_ITEM)

    context_pos = prompt.rfind(SIQA_ITEM["context"])
    question_pos = prompt.rfind(SIQA_ITEM["question"])
    assert context_pos != -1 and question_pos != -1
    assert context_pos < question_pos, "SIQA context should precede the question"


def test_siqa_backward_compatible_without_context() -> None:
    """Old-format rows (context merged into the question) must still render."""
    legacy = {
        "question": "Tracy didn't go home that evening. What does Tracy need to do before this?",
        "choices": SIQA_ITEM["choices"],
        "ground_truth": SIQA_ITEM["ground_truth"],
    }
    prompt = _render("siqa", legacy)

    assert legacy["question"] in prompt
    assert prompt.rstrip().endswith("Answer:")


# A trimmed RACE example: the passage lives in "context" and the question in
# "question", mirroring the updated dataset layout.
RACE_ITEM = {
    "id": 0,
    "context": (
        "I am a psychologist. I first met Timothy, a quiet, overweight eleven-year-old boy, "
        "when his mother brought him to me to discuss his declining grades."
    ),
    "question": "What did the writer think of Timothy after learning about his typical day?",
    "choices": [
        "Timothy was very hardworking.",
        "Timothy was being mistreated.",
        "Timothy had a heavy burden.",
        "Timothy was enjoying his childhood.",
    ],
    "ground_truth": "Timothy had a heavy burden.",
}


def test_race_prompt_includes_article_and_question() -> None:
    """RACE prompts must keep the article and emit a separate 'Question:' label."""
    prompt = _render("race", RACE_ITEM)

    assert RACE_ITEM["context"] in prompt, f"RACE prompt dropped the article:\n{prompt}"
    assert "Question:" in prompt, f"RACE prompt is missing a 'Question:' label:\n{prompt}"
    assert RACE_ITEM["question"] in prompt

    article_pos = prompt.rfind(RACE_ITEM["context"])
    question_pos = prompt.rfind(RACE_ITEM["question"])
    assert article_pos < question_pos, "RACE article should precede the question"


@pytest.mark.parametrize("task", ["siqa", "race"])
def test_context_tasks_do_not_drop_context(task: str) -> None:
    """Guard the core invariant: a provided context is never silently dropped."""
    item = SIQA_ITEM if task == "siqa" else RACE_ITEM
    prompt = _render(task, item)
    assert item["context"] in prompt
