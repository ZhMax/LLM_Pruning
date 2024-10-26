CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--resume_quant /home/exp_results/efficient_qat/e2e-qp-output/Llama-2-7b-w3g128-real-qp/checkpoint-128 \
--net Llama-2 \
--wbits 3 \
--group_size 128 \
--output_dir /home/exp_results/efficient_qat/e2e-qp-output/Llama-2-7b-w3g128-real-qp/inference_results/ \
--eval_tasks  winogrande,boolq,hellaswag,swag,xwinograd_en
