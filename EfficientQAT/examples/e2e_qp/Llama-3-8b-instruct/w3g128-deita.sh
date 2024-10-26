CUDA_VISIBLE_DEVICES=2 python main_e2e_qp.py \
    --quant_model_path ./output/block_ap_models/Llama-3-8b-instruct-w3g128 \
    --model_family llama3 \
    --wbits 3 \
    --group_size 128 \
    --learning_rate 1e-5 \
    --dataset deita-10k \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/Llama-2-7b-w3g128-deita \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --save_strategy epoch \
    --training_strategy epochs \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --max_train_samples 8192 \
    --num_train_epochs 1 \
    --eval_dataset_size 64 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --preprocessing_num_workers 32 \
    --do_ppl_eval