#!/bin/bash

cd /home/LLM_Pruning/Sparse-Expansion;
python main.py \
    --model llama \
    --path_to_model /home/LLaMA/huggingface/Llama-2-7b-hf \
    --output_path /home/exp_results/sparse_moe/llama7b_moe_sparse_2@4_test \
    --model_size 7B \
    --cache_dir /home/data/hf_cache \
    --sparsity 2:4 \
    --verbose --dataset wikitext2 --dataset-size 32 --num_clusters 2 --PCA_reduction_factor 32