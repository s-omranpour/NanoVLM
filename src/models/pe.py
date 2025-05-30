import math
import torch
from torch import nn
import torch.nn.functional as F


class NoPE(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x).to(x.device)

class LAPE(nn.Module):
    def __init__(self, max_size=512, d_model=128):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(max_size, d_model), requires_grad=True)

    def forward(self, x):
        bs, l, d = x.shape
        assert l <= self.max_size, "Input size exceeded max_size"
        assert d == self.d_model, "Invalid hidden size"
        return self.pe[:l].unsqueeze(0).repeat(bs, 1, 1)
        

class SinPE(nn.Module):
    def __init__(self, max_size=512, d_model=128):
        super().__init__()
        self.max_size = max_size
        self.d_model = d_model
        
        self.pe = nn.Parameter(torch.zeros(max_size, d_model), requires_grad=False)
        pos = torch.arange(0., max_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * (math.log(1.0/10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)
        
    def forward(self, x):
        bs, l, d = x.shape
        assert l <= self.max_size, "Input size exceeded max_size"
        assert d == self.d_model, "Invalid hidden size"
        return self.pe[:l].unsqueeze(0).repeat(bs, 1, 1)
        
        