from pathlib import Path

import numpy as np


def is_audio_input(inputs):
    """Return whether the input uses a supported audio suffix."""
    suffix = (
        Path(inputs).suffix if not isinstance(inputs, list) else Path(inputs[0]).suffix
    )
    return suffix in {".wav", ".flac", ".mp3"}


def filter_error_documents(corpus_ids, corpus_texts):
    """Remove corpus documents whose text contains an error marker."""
    valid_corpus_indices = [
        i for i, text in enumerate(corpus_texts) if "error" not in text.strip().lower()
    ]
    corpus_ids = [corpus_ids[i] for i in valid_corpus_indices]
    corpus_texts = [corpus_texts[i] for i in valid_corpus_indices]
    return corpus_ids, corpus_texts


def build_relevant_docs(qrels, corpus_ids):
    """Map each query to its available relevant corpus documents."""
    relevant_docs = {}
    corpus_id_set = set(corpus_ids)
    for row in qrels:
        qid, cid = row["query-id"], row["corpus-id"]
        if cid not in corpus_id_set:
            continue
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(cid)
    return relevant_docs


def compute_dcg_at_k(relevances, k):
    """Compute Discounted Cumulative Gain at k - matches official implementation."""
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg
