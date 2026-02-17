import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Set

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate NLLB-LLM2Vec model on text retrieval.")
    
    parser.add_argument("--model_name", type=str, 
                        default="fdschmidt93/NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse",
                        help="HuggingFace model name or path.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (for offline mode). If not specified, uses model_name.")
    parser.add_argument("--queries_path", type=str, required=True, 
                        help="Path to the queries dataset.")
    parser.add_argument("--corpus_path", type=str, required=True, 
                        help="Path to the corpus dataset.")
    parser.add_argument("--qrels_path", type=str, required=True, 
                        help="Path to the qrels dataset.")
    parser.add_argument("--query_lang", type=str, default='wol_Latn',
                        help="Source language code for queries (e.g., 'wol_Latn', 'eng_Latn'). None for English.")
    parser.add_argument("--document_lang", type=str, default='fra_Latn',
                        help="Source language code for documents/corpus (e.g., 'fra_Latn'). None for English.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding.")
    parser.add_argument("--output_dir", type=str, default="./eval_output", 
                        help="Directory for results.")
    
    return parser.parse_args()


class NLLBLLM2VecWrapper:
    """Wrapper for NLLB-LLM2Vec model."""
    
    def __init__(self, model_name: str, tokenizer_path: str = None, batch_size: int = 32):
        # Load tokenizer first
        tokenizer_path = tokenizer_path or model_name
        logging.info(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
        
        # Load model
        logging.info(f"Loading model from: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # CRITICAL: Force set the tokenizer BEFORE any encode call
        # This prevents the model from trying to download the hardcoded tokenizer
        self.model._tokenizer = self.tokenizer
        
        # Also patch the tokenizer property to always return our tokenizer
        type(self.model).tokenizer = property(lambda self: self._tokenizer)
        
        self.batch_size = batch_size
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, sentences, src_lang=None, batch_size=None):
        """Encode sentences to embeddings."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            if src_lang:
                embeddings = self.model.encode(batch, src_lang=src_lang)
            else:
                embeddings = self.model.encode(batch)
            
            all_embeddings.append(embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logging.info(f"Encoded {i + len(batch)}/{len(sentences)} sentences")
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(all_embeddings.shape)
        return all_embeddings


def compute_dcg_at_k(relevances, k):
    """Compute Discounted Cumulative Gain at k - matches official implementation."""
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg


def compute_metrics(similarity_scores: torch.Tensor, relevant_docs: Dict[str, Set[str]], 
                   query_ids: list, corpus_ids: list, k_values: list = [5, 10]):
    """
    Compute NDCG@k metric matching the official InformationRetrievalEvaluator implementation.
    """
    metrics = {}
    
    # Get max_k to retrieve top-k results once
    max_k = max(k_values)
    
    # Get top-k predictions for all queries
    top_k_scores, top_k_indices = torch.topk(
        similarity_scores, 
        k=min(max_k, similarity_scores.shape[1]), 
        dim=1, 
        largest=True, 
        sorted=True  # Important: sorted=True to get them in descending order
    )
    
    for k in k_values:
        ndcgs = []
        
        for i, query_id in enumerate(query_ids):
            if query_id not in relevant_docs:
                continue
                
            query_relevant_docs = relevant_docs[query_id]
            if len(query_relevant_docs) == 0:
                continue
            
            # Get top-k predictions for this query
            top_k_corpus_ids = [corpus_ids[idx] for idx in top_k_indices[i, :k].cpu().tolist()]
            
            # predicted_relevance: 1 if document is relevant, 0 otherwise
            predicted_relevance = [1 if corpus_id in query_relevant_docs else 0 for corpus_id in top_k_corpus_ids]
            
            # true_relevances: all relevant documents have relevance 1
            true_relevances = [1] * len(query_relevant_docs)
            
            # Compute NDCG@k
            ndcg_value = compute_dcg_at_k(predicted_relevance, k) / compute_dcg_at_k(true_relevances, k)
            ndcgs.append(ndcg_value)
        
        # Store metric
        metrics[f"ndcg@{k}"] = np.mean(ndcgs) if ndcgs else 0.0
    print(metrics)
    return metrics


def load_model(model_name: str, tokenizer_path: str, batch_size: int) -> NLLBLLM2VecWrapper:
    """Loads the NLLB-LLM2Vec model."""
    logging.info(f"Loading model: {model_name}")
    return NLLBLLM2VecWrapper(model_name=model_name, tokenizer_path=tokenizer_path, batch_size=batch_size)


def run_retrieval_evaluation(args, model: NLLBLLM2VecWrapper):
    """Sets up and runs the Information Retrieval evaluation."""
    logging.info("Starting Information Retrieval evaluation.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Loading datasets for retrieval...")
    try:
        queries_ds = load_dataset(args.queries_path, split="train")
        corpus_ds = load_dataset(args.corpus_path, split="train")
        qrels_ds = load_dataset(args.qrels_path, split="train")
    except:
        queries_ds = load_from_disk(args.queries_path)
        corpus_ds = load_from_disk(args.corpus_path)
        qrels_ds = load_from_disk(args.qrels_path)

    # Prepare queries (no prompt prepending)
    query_ids = queries_ds["_id"]
    query_texts = queries_ds["text"]
    
    # Prepare corpus
    corpus_ids = corpus_ds["_id"]
    corpus_texts = corpus_ds["text"]
    
    # Filter out corpus errors
    valid_corpus_indices = [i for i, text in enumerate(corpus_texts) 
                           if "error" not in text.strip().lower()]
    corpus_ids = [corpus_ids[i] for i in valid_corpus_indices]
    corpus_texts = [corpus_texts[i] for i in valid_corpus_indices]
    
    logging.info(f"Filtered corpus from {len(corpus_ds)} to {len(corpus_texts)} documents (removed errors).")
    
    # Prepare relevant docs
    relevant_docs = {}
    corpus_id_set = set(corpus_ids)
    for row in qrels_ds:
        qid, cid = row["query-id"], row["corpus-id"]
        if cid not in corpus_id_set:
            continue
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(cid)

    logging.info(f"Loaded {len(query_ids)} queries, {len(corpus_ids)} corpus documents, {len(relevant_docs)} query-document relationships.")

    # Encode queries
    logging.info("Encoding queries...")
    query_start = time.time()
    query_embeddings = model.encode(query_texts, src_lang=args.query_lang)
    query_time = time.time() - query_start
    logging.info(f"Query encoding completed in {query_time:.2f} seconds")
    
    # Encode corpus
    logging.info("Encoding corpus...")
    corpus_start = time.time()
    corpus_embeddings = model.encode(corpus_texts, src_lang=args.document_lang)
    corpus_time = time.time() - corpus_start
    logging.info(f"Corpus encoding completed in {corpus_time:.2f} seconds")
    
    # Compute similarity
    logging.info("Computing similarity scores...")
    query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)
    corpus_embeddings_norm = F.normalize(corpus_embeddings, p=2, dim=1)
    similarity_scores = query_embeddings_norm @ corpus_embeddings_norm.T
    
    # Compute metrics
    logging.info("Computing metrics...")
    metrics = compute_metrics(similarity_scores, relevant_docs, query_ids, corpus_ids, k_values=[5, 10])
    
    logging.info(f"\nNDCG Evaluation Results:")
    logging.info(f"{'='*60}")
    for metric, value in sorted(metrics.items()):
        logging.info(f"{metric:20s}: {value:.4f}")
    logging.info(f"{'='*60}")
    logging.info(f"Total evaluation time: {query_time + corpus_time:.2f} seconds")
    
    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write("NDCG Evaluation Results\n")
        f.write("="*60 + "\n")
        for metric, value in sorted(metrics.items()):
            f.write(f"{metric:20s}: {value:.4f}\n")
        f.write("="*60 + "\n")
        f.write(f"Query encoding time: {query_time:.2f} seconds\n")
        f.write(f"Corpus encoding time: {corpus_time:.2f} seconds\n")
        f.write(f"Total time: {query_time + corpus_time:.2f} seconds\n")
    
    logging.info(f"Results saved to {results_file}")
    
    return metrics


def main():
    """Main function to orchestrate the evaluation."""
    args = get_args()
    
    model = load_model(
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size
    )
    
    run_retrieval_evaluation(args, model)


if __name__ == "__main__":
    main()

# fleurs:
# ndcg@10: 0.5943
# ndcg@5: 0.5598

# kallama:
# ndcg@10: 0.6153
# ndcg@5: 0.5798