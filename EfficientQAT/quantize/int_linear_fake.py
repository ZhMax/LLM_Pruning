import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
        outlier_ids=None,
        training_mode='full' #'quant', 'outlier', 'full'
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        # initialize quantizer
        if outlier_ids is None:
            self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
        else:
            self.set_mask(outlier_ids)
            self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight[:, self.mask])
            self.training_mode = training_mode
        self.use_temporary_parameter = False

    @torch.no_grad
    def set_mask(self, outlier_ids: torch.tensor):
        self.mask = torch.ones(self.in_features, 
                                dtype=torch.bool)
        self.mask[outlier_ids] = False

        col_ids = torch.arange(self.in_features, 
                               dtype=torch.int32)
        col_perm = torch.cat([col_ids[self.mask], 
                                   col_ids[~self.mask]])

        self.inv_col_perm = torch.zeros(col_perm.numel(), 
                                        dtype=col_perm.dtype)
        self.inv_col_perm[col_perm] = torch.arange(col_perm.numel(),
                                                        dtype=col_perm.dtype)   
    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            if hasattr(self, 'mask'):

                q_weight = self.weight_quantizer(self.weight[:, self.mask])
                fp_weight = self.weight[:, ~self.mask]
                
                if self.training_mode == 'quant':
                    fp_weight = fp_weight.detach()
                elif self.training_mode == 'outlier':
                    q_weight = q_weight.detach()
                elif self.training_mode != 'full':
                    raise ValueError("training mode is not correct! Try 'quant', 'outlier', or 'full'")
                
                weight = torch.hstack([q_weight, fp_weight])
                weight = weight[:, self.inv_col_perm]
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant




