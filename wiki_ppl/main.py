import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from importlib.metadata import version
from lib.eval import eval_ppl, eval_zero_shot

from datetime import datetime

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    # model.seqlen = model.config.max_position_embeddings 
    model.seqlen = int(model.config.max_position_embeddings // 2)
    return model

def save_logs(model_name, ppl_test, path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    cur_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_filepath = os.path.join(path_to_save, f"log_{cur_datetime}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{model_name}\t{ppl_test:.4f}", file=f, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--peft', type=str, default=None, help='Path to PEFT adapters')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="/scratch/llm_weights", type=str )
    parser.add_argument('--save_logs', type=str, default=None, help='Path to save results.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Loading model and tokenizer
    model_name = args.model.split("/")[-1]
    model = get_llm(args.model, args.cache_dir)

    if args.peft is not None:
        model = PeftModel.from_pretrained(model, args.peft)

    model.eval()
    print(model.hf_device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Set device
    device = torch.device("cuda:0")
    print("use device ", device)

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.save_logs:
        save_logs(model_name, ppl_test, args.save_logs)

if __name__ == '__main__':
    main()



