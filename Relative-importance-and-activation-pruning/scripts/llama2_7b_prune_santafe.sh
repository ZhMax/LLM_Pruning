#!/bin/bash

python /home/LLM_Pruning/Relative-importance-and-activation-pruning/main.py \
    --model /home/LLaMA/huggingface/Llama-2-7b-hf \
	--prune_method ria \
    --calib_dataset c4 \
	--sparsity_ratio 0.5 \
	--sparsity_type 2:4 \
    --reallocation \
    --lsa \
    --save_model /home/exp_results/ria_pruning/llama-2-7b-ria-2@4