import torch
from transformers import AutoModelForCausalLM
from contextlib import contextmanager

@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def get_model(model_path, hf_token=None, trust_remote_code=True):

    with suspend_nn_inits():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            torch_dtype=torch.bfloat16, 
            use_auth_token=hf_token, 
            low_cpu_mem_usage=True
        )
        
    print("Model loaded sucсessfully ...")

    return model


def find_blocks(module, blocks=['mlp'], current_key_name=None, res=None):
    
    if res is None:
        res = []

    for name, module in module.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if current_key_name[-1] in blocks:
            res.append('.'.join(current_key_name))
    
        if len(list(module.children())) > 0:
            res = find_blocks(
                module, 
                blocks=blocks, 
                current_key_name=current_key_name,
                res=res
            )
        
        current_key_name.pop(-1)
    
    return res