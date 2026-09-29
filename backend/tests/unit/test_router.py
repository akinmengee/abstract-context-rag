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
    "What are the main findings of this paper?",
    "TL;DR?",
    "tldr this paper",
    "In a nutshell, what does this paper say?",
    "What's the gist of this paper?",
    "Give me a high-level overview.",
    # Measured live (2026-09-27): a Wikipedia article isn't a "paper", so a
    # question using its own vocabulary needs to route the same way.
    "What is this research about?",
    "What is this source about?",
    "Summarize this text.",
    "What does this content cover?",
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
    # "main"/"key" phrases are not anchored to the document (see router.py),
    # so a nearby table/figure reference is what has to catch this instead.
    "What key finding does Table 3 report about retrieval accuracy?",
    "What is the main result shown in Table 1?",
    # The new _DOC synonyms shouldn't fire on unrelated uses of the same words.
    "What research methods does this paper use?",
    "Where does this text mention the learning rate?",
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


def test_a_trigger_word_does_not_leak_into_an_unrelated_next_sentence():
    # "summary" and "this paper" are each real, just in different sentences.
    assert not is_global_question(
        "I already read the summary of related work. "
        "Does this paper also evaluate on TriviaQA?"
    )
