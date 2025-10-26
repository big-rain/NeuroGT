import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from model.base_MHA import BiasedMHA
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from model.T_layer import GraphormerLayer


class GTConv(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 4,
            dropout: float = 0.1,
            act: str = 'gelu',
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: str = 'layer_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            attn_type: str = 'multihead',
            attn_kwargs: Optional[Dict[str, Any]] = None,
            fusion_type: str = 'gated',  # 新增融合机制选项
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type
        self.fusion_type = fusion_type
        act_kwargs = act_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        # ====================
        # 1. 全局注意力模块
        # ====================
        if attn_type == 'graphormer':
            self.attn = GraphormerLayer(
                feat_size=channels,
                num_heads=heads,
                attn_bias_type='add',
                norm_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'multihead':
            self.attn = nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                dropout=dropout,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        elif attn_type == 'baseMHA':
            self.attn = BiasedMHA(
                feat_size=channels,
                num_heads=heads,
                attn_bias_type='add',
                attn_drop=dropout,
            )

        else:
            raise ValueError(f'不支持的注意力类型: {attn_type}')

        # ====================
        # 2. 特征增强模块
        # ====================
        self.activation = self._get_activation(act, **act_kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

        # ====================
        # 3. 归一化层
        # ====================
        self.norm_local = self._get_norm(norm, channels, **norm_kwargs)
        self.norm_global = self._get_norm(norm, channels, **norm_kwargs)
        self.norm_output = self._get_norm(norm, channels, **norm_kwargs)

        # ====================
        # 4. 特征融合机制
        # ====================
        if fusion_type == 'gated':
            self.fusion_gate = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.Sigmoid()
            )
        elif fusion_type == 'attention':
            self.fusion_attn = nn.MultiheadAttention(channels, 1, batch_first=True)
        elif fusion_type == 'weighted':
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数"""
        # 局部卷积初始化
        if self.conv is not None:
            if hasattr(self.conv, 'reset_parameters'):
                self.conv.reset_parameters()

        # 注意力模块初始化
        if hasattr(self.attn, 'reset_parameters'):
            self.attn.reset_parameters()
        elif hasattr(self.attn, '_reset_parameters'):
            self.attn._reset_parameters()

        # MLP初始化
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # 归一化层初始化
        for norm in [self.norm_local, self.norm_global, self.norm_output]:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

        # 融合门初始化
        if hasattr(self, 'fusion_gate'):
            nn.init.xavier_uniform_(self.fusion_gate[0].weight)
            nn.init.constant_(self.fusion_gate[0].bias, 0)

    def _get_activation(self, act: str, **kwargs):
        """获取激活函数实例"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(**kwargs),
            'elu': nn.ELU(**kwargs),
            'silu': nn.SiLU()
        }
        return activations.get(act.lower(), nn.GELU())

    def _get_norm(self, norm: str, channels: int, **kwargs):
        """获取归一化层实例"""
        if norm.lower() == 'batch_norm':
            return nn.BatchNorm1d(channels, **kwargs)
        elif norm.lower() == 'layer_norm':
            return nn.LayerNorm(channels, **kwargs)
        elif norm.lower() == 'instance_norm':
            return nn.InstanceNorm1d(channels, **kwargs)
        return nn.Identity()

    def _fuse_features(self, local_feat, global_feat):
        """融合局部和全局特征"""
        if self.fusion_type == 'gated':
            # 门控融合
            combined = torch.cat([local_feat, global_feat], dim=-1)
            gate = self.fusion_gate(combined)
            return gate * local_feat + (1 - gate) * global_feat

        elif self.fusion_type == 'attention':
            # 注意力融合
            query = local_feat.unsqueeze(1)
            key = global_feat.unsqueeze(1)
            value = global_feat.unsqueeze(1)
            fused, _ = self.fusion_attn(query, key, value)
            return fused.squeeze(1)

        elif self.fusion_type == 'weighted':
            # 加权融合
            alpha = torch.sigmoid(self.fusion_weight[0])
            beta = torch.sigmoid(self.fusion_weight[1])
            return alpha * local_feat + beta * global_feat

        else:  # 默认求和
            return local_feat + global_feat

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            attn_bias: Optional[Tensor] = None,
            batch: Optional[Tensor] = None,
            **kwargs
    ) -> Tensor:
        # ===================================
        # 1. 局部图卷积 (Local MPNN)
        # ===================================
        local_feat = x
        # print(local_feat)
        if self.conv is not None:
            local_feat = self.conv(x, edge_index, **kwargs)
            local_feat = F.dropout(local_feat, p=self.dropout, training=self.training)

            # 残差连接 + 归一化
            local_feat = local_feat + x
            local_feat = self.norm_local(local_feat)

        # ===================================
        # 2. 全局注意力 (Global Attention)
        # ===================================
        # 转换为密集表示

        dense_x, mask = to_dense_batch(x, batch)

        # 应用注意力机制
        if self.attn_type == 'graphormer':
            # print(dense_x)
            global_feat = self.attn(dense_x, attn_bias=attn_bias, attn_mask=mask)
            # global_feat = self.attn(dense_x,  attn_mask=mask)
        elif self.attn_type == 'multihead':
            global_feat, _ = self.attn(
                dense_x, dense_x, dense_x,
                key_padding_mask=~mask,
                need_weights=False
            )
        elif isinstance(self.attn, BiasedMHA):
            global_feat = self.attn(dense_x, attn_bias=attn_bias, attn_mask=mask)

        elif self.attn_type == 'performer':
            global_feat = self.attn(dense_x, mask=mask)

        # 转换回稀疏表示
        global_feat = global_feat[mask]
        # 残差连接 + 归一化
        global_feat = F.dropout(global_feat, p=self.dropout, training=self.training)
        global_feat = global_feat + x
        global_feat = self.norm_global(global_feat)

        # ===================================
        # 3. 特征融合
        # ===================================
        fused = self._fuse_features(local_feat, global_feat)

        # ===================================
        # 4. 特征增强
        # ===================================
        out = fused + self.mlp(fused)
        out = self.norm_output(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')
