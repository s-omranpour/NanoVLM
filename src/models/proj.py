import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjector(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        mode='pool',
        factor=1, 
        pixel_shuffle_factor=1
    ):
        super().__init__()
        assert mode in ['pool', 'shuffle']
        self.mode = mode
        self.factor = factor
        if mode == 'pool':
            self.conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=factor, stride=factor)
        else:
            self.lin = nn.Linear(input_dim*(factor**2), hidden_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.factor
        w_out = width // self.factor
        
        x = x.reshape(bsz, h_out, self.factor, w_out, self.factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.factor**2)
        return x
    
    def forward(self, x):
        # x: [B, C, H, W]
        if self.mode == 'shuffle':
            h = self.pixel_shuffle(
                x.flatten(2).permute(0, 2, 1)
            )
            return self.lin(h)

        h = self.conv(x)
        return h.flatten(2).permute(0, 2, 1)