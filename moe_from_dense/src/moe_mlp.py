import importlib
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class MoeMLP(nn.Module):

    def __init__(
            self, 
            hidden_size, 
            intermediate_size,
            model_config, 
            num_local_experts, 
            num_experts_per_tok
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = ACT2FN[model_config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj