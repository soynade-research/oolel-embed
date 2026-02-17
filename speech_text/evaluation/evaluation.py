import argparse
import logging
import shutil
import string
import time
from pathlib import Path
from uuid import uuid4

import soundfile as sf
import torch
from datasets import Dataset, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..models.oolel_embed import OolelEmbed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPTS = {
    "intent": "Given a Wolof speech, detect its intent\nQuery:",
    "topic": "Given a Wolof speech, classify its topic\nQuery:",
    "translation": "Given a Wolof query, retrieve the French texts that translate the query\nQuery:",
    "transcription": "Given a Wolof speech query, retrieve the Wolof texts that transcribe the speech query\nQuery:",
    "document": "Given a speech Wolof query, retrieve relevant passages that answer the speech query\nQuery:",
    "text_document": "Given a Wolof query, retrieve relevant passages that answer the query\nQuery:",
    "text_intent": "Given a Wolof query, detect its intent\nQuery:",
}

TABLE = str.maketrans({p: " " for p in string.punctuation})


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a speech and text embedding model.")
    
    parser.add_argument("--text_embedding_model_path", type=str, required=True, help="Path to the text embedding model.")
    parser.add_argument("--speech_embedding_model_path", type=str, required=True, help="Path to the speech embedding model.")
    parser.add_argument("--model_mode", type=str, default="lf", choices=["clip", "lf"], help="The model mode.")
    parser.add_argument("--dimension", type=int, required=True, help="The embedding dimension to use.")
    parser.add_argument("--prompt_key", type=str, default="transcription", choices=PROMPTS.keys(), help="Key for the prompt to use.")
    parser.add_argument("--output_dir", type=str, default="./eval_output", help="Directory for temporary files and results.")
    parser.add_argument("--text_only", action="store_true", help="Run evaluation in text-only mode.")
    
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Evaluation mode")

    retrieval_parser = subparsers.add_parser("retrieval", help="Run information retrieval evaluation.")
    retrieval_parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries dataset.")
    retrieval_parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus dataset.")
    retrieval_parser.add_argument("--qrels_path", type=str, required=True, help="Path to the qrels dataset.")

    classification_parser = subparsers.add_parser("classification", help="Run zero-shot classification evaluation.")
    classification_parser.add_argument("--dataset_path", type=str, required=True, help="Path to the classification dataset.")
    classification_parser.add_argument("--query_column", type=str, default="audio", help="Name of the column containing the query.")
    classification_parser.add_argument("--label_column", type=str, default="label", help="Name of the column containing the labels.")

    return parser.parse_args()


def save_wav_file(sample, wav_dir: Path):
    """Saves a single audio sample to a .wav file."""
    wav_dir.mkdir(exist_ok=True, parents=True)
    audio_id = str(uuid4())
    filepath = wav_dir / f"{audio_id}.wav"
    
    sf.write(
        filepath,
        sample["audio"]["array"],
        samplerate=sample["audio"]["sampling_rate"]
    )
    sample["audio_query_path"] = str(filepath)
    return sample


class ClassificationEvaluator:
    """Evaluates a model on a zero-shot classification task."""
    def __init__(self, model: SentenceTransformer, dataset: Dataset, query_column: str, label_column: str):
        self.model = model
        self.dataset = dataset
        self.query_column = query_column
        self.label_column = label_column
        self.labels = sorted(list(set(dataset[label_column])))

    def __call__(self, output_dir: Path, text_only: bool, prompt=""):
        """Runs the evaluation and prints metrics."""
        logging.info("Preparing classification data...")
        wav_dir = output_dir / "wavs"
        if wav_dir.exists():
            shutil.rmtree(wav_dir)

        if not text_only:
            self.dataset = self.dataset.map(save_wav_file, fn_kwargs={"wav_dir": wav_dir}, keep_in_memory=True)
            queries = self.dataset["audio_query_path"]
        else:
            prompt = PROMPTS.get(prompt, "")
            self.dataset = self.dataset.map(lambda x: {"text_query": f"{prompt} {x[self.query_column]}"})
            queries = self.dataset["text_query"]

        logging.info(f"Encoding {len(self.labels)} labels...")
        labels_embed = self.model.encode_document(self.labels)
        
        logging.info(f"Encoding {len(queries)} queries...")
        if text_only:
            queries_embed = self.model.encode_document(queries)
        else:
            queries_embed = self.model.encode_query(queries)

        logging.info("Calculating similarity and predicting classes...")
        similarity = self.model.similarity(queries_embed, labels_embed)
        classifications = similarity.argmax(dim=-1)
        
        golds = self.dataset[self.label_column]
        predicted = [self.labels[idx] for idx in classifications]

        self._print_metrics(golds, predicted)

    def _print_metrics(self, golds, predicted):
        """Calculates and prints classification metrics."""
        accuracy = accuracy_score(golds, predicted)
        f1 = f1_score(golds, predicted, average='weighted', zero_division=0)
        recall = recall_score(golds, predicted, average='weighted', zero_division=0)
        precision = precision_score(golds, predicted, average='weighted', zero_division=0)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score (Weighted): {f1:.4f}")
        logging.info(f"Recall (Weighted): {recall:.4f}")
        logging.info(f"Precision (Weighted): {precision:.4f}")


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def load_model(text_model_path: str, speech_model_path: str, prompt: str, model_mode, dimension: int=1024) -> SentenceTransformer:
    """Loads the OolelEmbed model wrapped in a SentenceTransformer."""
    logging.info("Loading model...")
    model = OolelEmbed(
        text_embedding_model_path=text_model_path,
        speech_embedding_model_path=speech_model_path,
        prompt=prompt,
        mode=model_mode,
        truncate_dim=dimension
    )
    return SentenceTransformer(modules=[model])

@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_retrieval_evaluation(args, model: SentenceTransformer):
    """Sets up and runs the Information Retrieval evaluation."""
    logging.info("Starting Information Retrieval evaluation.")
    
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    if wav_dir.exists():
        shutil.rmtree(wav_dir)

    logging.info("Loading datasets for retrieval...")
    try:
        queries_ds = load_dataset(args.queries_path, split="train")
        corpus_ds = load_dataset(args.corpus_path, split="train")
        qrels_ds = load_dataset(args.qrels_path, split="train")
    except:
        queries_ds = load_from_disk(args.queries_path)
        corpus_ds = load_from_disk(args.corpus_path)
        qrels_ds = load_from_disk(args.qrels_path)

    if args.text_only:
        prompt = PROMPTS.get(args.prompt_key, "")
        queries_ds = queries_ds.map(lambda x: {"processed_query": f"{prompt} {x['text']}"})
        queries = dict(zip(queries_ds["_id"], queries_ds["processed_query"]))
    else:
        queries_ds = queries_ds.map(save_wav_file, fn_kwargs={"wav_dir": wav_dir}, keep_in_memory=True)
        queries = dict(zip(queries_ds["_id"], queries_ds["audio_query_path"]))
    
    corpus = dict(zip(corpus_ds["_id"], corpus_ds["text"]))
    
    corpus_errors = {row["_id"] for row in corpus_ds if "error" in row.get("text", "").strip().lower()}
    
    relevant_docs = {}
    for row in qrels_ds:
        qid, cid = row["query-id"], row["corpus-id"]
        if cid in corpus_errors:
            continue
        if qid not in relevant_docs:
            relevant_docs[qid] = set()
        relevant_docs[qid].add(cid)

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="retrieval-test",
        accuracy_at_k=[5, 10],
        ndcg_at_k=[5, 10],
    )

    logging.info("Running evaluator...")
    start_time = time.time()
    results = ir_evaluator(model)
    end_time = time.time()
    
    logging.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    print(results)


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_classification_evaluation(args, model: SentenceTransformer):
    """Sets up and runs the Zero-Shot Classification evaluation."""
    logging.info("Starting Zero-Shot Classification evaluation.")
    
    try:
        dataset = load_dataset(args.dataset_path, split="train")
    except:
        dataset = load_from_disk(args.dataset_path)
    evaluator = ClassificationEvaluator(
        model=model.cuda(),
        dataset=dataset,
        query_column=args.query_column,
        label_column=args.label_column
    )
    
    evaluator(output_dir=Path(args.output_dir), text_only=args.text_only, prompt=args.prompt_key)


def main():
    """Main function to orchestrate the evaluation."""
    args = get_args()
    
    model = load_model(
        text_model_path=args.text_embedding_model_path,
        speech_model_path=args.speech_embedding_model_path,
        prompt=PROMPTS[args.prompt_key],
        model_mode=args.model_mode,
        dimension=args.dimension
    )

    if args.mode == "retrieval":
        run_retrieval_evaluation(args, model)
    elif args.mode == "classification":
        run_classification_evaluation(args, model)
    else:
        logging.error(f"Unknown evaluation mode: {args.mode}")


if __name__ == "__main__":
    main()