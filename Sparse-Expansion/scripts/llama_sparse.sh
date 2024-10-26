#!/bin/bash

cd /home/LLM_Pruning/Sparse-Expansion;
python main.py --model llama --path_to_model /home/llm_compression/LLaMA/Llama-2-7b-hf --output_path /home/llm_compression/Quantization/Sparse_moe/llama7b_moe_sparse_2@4_v1 --model_size 7B --cache_dir /home/data/hf_cache --sparsity 2:4 --verbose --dataset wikitext2 --dataset-size 128 --num_clusters 16 --PCA_reduction_factor 32