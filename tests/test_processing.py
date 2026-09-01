import numpy as np
import pytest

from speech_text.processing import (
    build_relevant_docs,
    compute_dcg_at_k,
    filter_error_documents,
    is_audio_input,
)


def test_audio_input_routing_keeps_existing_suffix_logic():
    assert is_audio_input("query.wav")
    assert is_audio_input(["first.mp3", "second.wav"])
    assert not is_audio_input("plain text")
    assert not is_audio_input("QUERY.WAV")


def test_error_documents_are_removed_without_changing_order():
    ids, texts = filter_error_documents(
        ["d1", "d2", "d3"],
        ["First", "ERROR: failed", "Third"],
    )
    assert ids == ["d1", "d3"]
    assert texts == ["First", "Third"]


def test_relevant_docs_ignore_removed_corpus_ids_and_deduplicate():
    qrels = [
        {"query-id": "q1", "corpus-id": "d1"},
        {"query-id": "q1", "corpus-id": "d1"},
        {"query-id": "q1", "corpus-id": "removed"},
    ]
    assert build_relevant_docs(qrels, ["d1"]) == {"q1": {"d1"}}


@pytest.mark.parametrize(
    ("relevances", "k", "expected"),
    [
        ([], 5, 0),
        ([1, 0, 1], 3, 1.5),
        ([1, 1, 1], 2, 1 + 1 / np.log2(3)),
    ],
)
def test_compute_dcg_at_k(relevances, k, expected):
    assert compute_dcg_at_k(relevances, k) == pytest.approx(expected)
