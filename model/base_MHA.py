"""Biased Multi-head Attention"""

import torch as th
import torch.nn as nn
import torch
import torch.nn.functional as F


class BiasedMHA(nn.Module):

    def __init__(
        self,
        feat_size,
        num_heads,
        bias=True,
        attn_bias_type="add",
        attn_drop=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size)
        self.k_proj = nn.Linear(feat_size, feat_size)
        self.v_proj = nn.Linear(feat_size, feat_size)
        self.o_proj = nn.Linear(feat_size, feat_size)
        self.attn_drop = nn.Dropout(attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of projection matrices, the same settings as in
        the original implementation of the paper.
        """
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-0.5)

        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.constant_(self.o_proj.bias, 0.0)

    def forward(self, x, attn_bias=None, attn_mask=None):

        B, N, C = x.size()
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = q * self.scale  # scaled dot-product

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, N]


        if attn_bias is not None:
            # print(">> [DEBUG] attn_bias NaN:", torch.isnan(attn_bias).any())
            if self.attn_bias_type == "add":
                attn_scores  += attn_bias
            else:
                attn_scores  *= attn_bias

        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [B, N]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            elif attn_mask.dim() == 3:  # [B, N, N]
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N, N]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Stability trick: subtract max
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        # --------------------------------------

        out = torch.matmul(attn_weights, v)  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.o_proj(out)
        return out

if __name__ == '__main__':
    feat_size = 128
    num_heads = 8
    bias = True
    attn_bias_type = "add"
    attn_drop = 0.1
    ndata = th.rand(16, 100, feat_size)
    attn_bias = th.rand(16, 100, 100, num_heads)
    attn_mask = th.rand(16, 100, 100) < 0.5

    net = BiasedMHA(feat_size, num_heads, bias, attn_bias_type, attn_drop)
    out = net(ndata, attn_bias, attn_mask)

    print(out.shape == (16, 100, feat_size))