import torch
from accelerate import init_empty_weights

from moetools.modelutils import get_model
from moetools.moe_mlp import MoeMLP


def make_moe_blocks(
    model, 
    model_config,
    num_local_experts=8, 
    num_experts_per_tok=2,
    current_key_name=None,
    block = ["mlp"]
):
    
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if current_key_name[-1] in block:
            with init_empty_weights():
                hidden_size = module.hidden_size
                intermediate_size = module.intermediate_size

                new_block = MoeMLP(
                    hidden_size=hidden_size, 
                    intermediate_size=intermediate_size,
                    model_config=model_config,
                    num_local_experts=num_local_experts, 
                    num_experts_per_tok=num_experts_per_tok
                )

            model_block = model._modules[name]
            new_block.load_state_dict(model_block.state_dict(), assign=True)
            model._modules[name] = new_block
        
        if len(list(module.children())) > 0:
            _ = make_moe_blocks(
                module,
                model_config=model_config,
                num_local_experts=num_local_experts,
                num_experts_per_tok=num_experts_per_tok,
                current_key_name=current_key_name
            )
        current_key_name.pop(-1)
    
    return model


@torch.no_grad()
def llm_sequential(
    model,
    num_local_experts,
    num_experts_per_tok
):
    print("Starting")

    print("Insert MoE blocks into the model...")
    model = make_moe_blocks(
        model,
        model_config=model.config,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok
    )

    model.config.quantization_config = {
        "quant_method": "moe",
        "num_local_experts": num_local_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "mlp_blocks_not_to_replace": None
    }

    print("MoE blocks have been inserted...")

    return model


def create_moe_model(args):
    model = get_model(args.model_path)
    
    model = llm_sequential(
        model, 
        args.num_local_experts,
        args.num_experts_per_tok
    )

    model.save_pretrained(args.save_path)
    print(f"Model has been saved in {args.save_path}")
