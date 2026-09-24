"""Query routing: deciding whether a question needs the whole document."""

import pytest

from abstractrag.rag.query.router import is_global_question

GLOBAL_QUESTIONS = [
    "Summarize this paper.",
    "Can you summarize the document?",
    "Please summarise the article for me.",
    "Give me an overview of this paper.",
    "What is this paper about?",
    "What is this document about?",
    "What does this paper cover?",
    "What is the main contribution of this paper?",
    "What are the key findings?",
    "What are the main takeaways?",
    "TL;DR?",
    "tldr this paper",
    "In a nutshell, what does this paper say?",
    "What's the gist of this paper?",
    "Give me a high-level overview.",
]

SPECIFIC_QUESTIONS = [
    "What retriever does this paper use?",
    # Contains "main" and "this paper" but is not a global question - a
    # single chunk can answer it, unlike "main contribution".
    "What is the main dataset used in this paper?",
    "How is the model trained?",
    "What is the learning rate?",
    "Which model is used as the generator component?",
    "How many parameters does BART have?",
    "What datasets does this paper evaluate on?",
    "How are the retriever and the generator trained?",
    # These used to be false positives: "summary"/"overview"/"key results" as
    # bare words matched even when the question is about one table or figure,
    # not the whole document.
    "What summary statistic does Table 3 report?",
    "What key results are in Table 2 for NQ?",
    "Explain the overview diagram in Figure 2.",
    "Give an overview of the DPR retriever architecture.",
]


@pytest.mark.parametrize("question", GLOBAL_QUESTIONS)
def test_global_questions_are_detected(question):
    assert is_global_question(question)


@pytest.mark.parametrize("question", SPECIFIC_QUESTIONS)
def test_specific_questions_are_not_flagged_as_global(question):
    assert not is_global_question(question)


def test_matching_is_case_insensitive():
    assert is_global_question("SUMMARIZE THIS PAPER")


def test_empty_question_is_not_global():
    assert not is_global_question("")
