import argparse
import re


def sparsity_type(value):
    # Check if the value is a float
    try:
        return float(value)
    except ValueError:
        pass

    # Check if the value is in the format N:M where N and M are integers and N <= M
    match = re.match(r"^(\d+):(\d+)$", value)
    if match:
        n, m = int(match.group(1)), int(match.group(2))
        # print("Regex match", n,m)
        if n <= m:
            return (n, m)

    raise argparse.ArgumentTypeError(
        "Sparsity must be a float or in the format N:M where N and M are integers and N <= M"
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="Argument parser with multiple groups"
    )

    # MoE parameters group
    moe_group = parser.add_argument_group(
        "Mixture of Experts Parameters Arguments"
    )
    moe_group.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model"
    )
    moe_group.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the model"
    )
    moe_group.add_argument(
        "--num_local_experts",
        type=int,
        required=False,
        default=8,
        help="Number of experts in each MLP block"
    )    
    moe_group.add_argument(
        "--num_experts_per_tok",
        type=int,
        required=False,
        default=2,
        help="Number of experts to route per-token"
    )

    return parser


