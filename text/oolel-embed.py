from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.util import cos_sim
import os
from pathlib import Path

PROMPTS = {
    "translation_retrieval": "Given a Wolof query, retrieve the French texts that translate the query\nQuery:",
    "topic_classification": "Given a Wolof text, classify its topic\nQuery:",
    "document_retrieval": "Given a Wolof query, retrieve relevant passages that answer the query\nQuery:",

    "speech_translation_retrieval": "Given a Wolof speech query, retrieve the French texts that translate the speech query\nQuery:",
    "speech_transcription_retrieval": "Given a Wolof speech query, retrieve the Wolof texts that transcribe the speech query\nQuery:",
    "speech_document_retrieval": "Given a speech Wolof query, retrieve relevant passages that answer the speech query\nQuery:",
    "speech_topic_classification": "Given a Wolof speech, classify its topic\nQuery:"
}

def train_model(model_path, dataset_path, output_dir):
    try:
        dataset = load_dataset(dataset_path, split="train")
    except:
        dataset = load_from_disk(dataset_path)
    dataset = dataset.map(
        lambda x: {"query": f"{PROMPTS[x['task_type']]} {x['query']}"},
        num_proc=32)
    print(dataset[0])
    # dataset = dataset.add_column("id", list(range(len(dataset))))
    dataset = dataset.rename_columns({"query": "anchor", "document": "positive"})
    dataset = dataset.train_test_split(test_size=0.005)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    matryoshka_dimensions = [1024, 512, 256, 128]

    ids = list(range(len(eval_dataset)))
    corpus = dict(
        zip(ids, eval_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(ids, eval_dataset["anchor"])
    )  # Our queries (qid => question)
    
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)
    
    evaluator = SequentialEvaluator(matryoshka_evaluators)

    model = SentenceTransformer(
        model_path,
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Oolel Embed",
        )
    )
    model.max_seq_length = 2048
    model.config = AutoConfig.from_pretrained(model_path)

    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=16,
        gradient_checkpointing=False,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=True,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=200,
        eval_steps=8_000,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=False,
        save_safetensors=False,
        # DeepSpeed configuration
        deepspeed="./deepspeed_config.json",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(
            ["anchor", "positive"]
        ),
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train(resume_from_checkpoint=bool(list(Path(output_dir).glob("checkpoint-*"))))

    model.save_pretrained(f"{output_dir}/final", safe_serialization=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--output_dir", type=str, default="models/oolel-text-embed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )