model="/home/LLaMA/huggingface/Llama-2-7b-hf"
method="wandg"
type="2:4"
suffix="weightonly"
save_dir="/home/exp_results/snip/llama2-7b-align-2@4-wiki-wandg"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data wikitext \
    --nsamples 64 \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --save_model $save_dir