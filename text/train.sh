#!/bin/bash

# DeepSpeed training script for 8-GPU setup
# Launch training with DeepSpeed
deepspeed --num_gpus=4 \
    --num_nodes=1 \
    oolel-embed.py \
    --model_path "Qwen/Qwen3-Embedding-0.6B" \
    --dataset_path "data/wo-fr-emb-2048/all_text_retrieval_data/" \
    --output_dir "models/oolel-text-embed"
