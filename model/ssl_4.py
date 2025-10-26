import copy

import torch


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 16:49
# @Author  : ShengPengpeng
# @File    : byol.py
# @Description :

import sys

sys.path.append("..")
from thop import profile
from engine.lr_scheduler import CosineDecayScheduler

import numpy as np
from dataloader.augment import MultiViewDataInjector, RandomJitterNeurite, NewBranchCutTransform

from dataloader.ACT import ACT
import os
import copy
import inspect
from typing import Any, Dict, Optional

import json
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

import torch_geometric.transforms as T
from torch_geometric.typing import Adj
from torch_geometric.nn.inits import reset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GINEConv, ResGatedGraphConv, global_mean_pool
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from model.PE_Encoder import HeatKernelEncoder
from model.base_MHA import BiasedMHA
from dataloader.HKPE import AddHeatKernelPE_v2
from dataloader.SPE import AddShortestPathPE
from dataloader.loader import NeuronDataset

import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class MLP_Predictor(nn.Module):
    r"""MLP used for predictor, The MLP has one hidden layer

    Args:
        input size (int) : size of input features
        output size (int) : size of output features
        hidden_size (int, optional): size of hidden layer (default : : object: '4096'
    """

    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class GG_layer(nn.Module):
    def __init__(self,
                 channels: int,
                 conv: Optional[MessagePassing],
                 heads:  int = 4,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.5,
                 act: str = 'relu',
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 norm: Optional[str] = 'batch_norm',
                 norm_kwargs: Optional[Dict[str, Any]] = None,

                 ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.attn = BiasedMHA(feat_size=channels,num_heads=heads,attn_bias_type='add',attn_drop=dropout)
        self.input_norm = nn.LayerNorm(channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                attn_bias,
                batch: Optional[torch.Tensor] = None,
                **kwargs
                ) -> Tensor:

        hs = []
        # Local Feature MPNN
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # global attention
        h, mask = to_dense_batch(x, batch)
        # h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = self.attn(h, attn_bias=attn_bias, attn_mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x     # residual connection

        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        # combine local and global outputs
        hs.append(h)
        out = sum(hs)

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, conv={self.conv}, heads={self.heads})')

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn.reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()


class Transformer_encoder(nn.Module):
    r"""  MPNN + Transformer """
    def __init__(self,
                 in_features: int,
                 channels: int,
                 pe_dim: int,
                 num_layer: int,
                 heads: int,
                 dropout: float,
                 attn_dropout: float,
                 ):
        super().__init__()

        # 预处理
        self.node_emb = nn.Linear(in_features, channels - pe_dim)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)
        self.edge_emb = nn.Embedding(32, channels)
        self.bias_encoder = HeatKernelEncoder(
            min_val=1e-5,
            max_val=1.0,
            num_bins=64,
            num_heads=8,
            log_transform=True
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layer):
            local_gnn = ResGatedGraphConv(in_channels=channels, out_channels=channels, act=nn.ReLU(), edge_dim=channels)
            global_nn = GG_layer(channels, local_gnn, heads=heads, dropout=dropout, attn_dropout=attn_dropout)
            self.convs.append(global_nn)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
        )

    def forward(self,data):
        x, pe, edge_index, edge_attr, batch = data.pos, data.pe, data.edge_index, data.edge_attr, data.batch

        x_pe = self.pe_norm(pe)
        pe_lin = self.pe_lin(x_pe)

        node_emb = self.node_emb(x)

        x = torch.cat((node_emb, pe_lin), dim=1)

        edge_attr = edge_attr.squeeze(-1).long()
        edge_attr = self.edge_emb(edge_attr)
        bias = getattr(data, 'bias', None)
        # attn_bias = self.bias_encoder(bias, batch=batch)
        if bias is not None:
            bias = bias
            attn_bias = self.bias_encoder(bias, batch=batch)
        else:
            attn_bias = None

        for conv in self.convs:
            node = conv(x, edge_index, attn_bias=attn_bias, batch=batch, edge_attr=edge_attr)
        x = global_mean_pool(node, batch)

        x = self.mlp(x)  # 2024.3.7 改

        return x, node

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for m in self.convs:
            m.reset_parameters()


class CoordNoise(nn.Module):
    """用于生成坐标噪声"""
    def __init__(self, noise_std=0.02, learnable=False):
        super().__init__()
        if learnable:
            self.noise_std = nn.Parameter(torch.tensor(noise_std))
        else:
            self.register_buffer("noise_std", torch.tensor(noise_std))

    def forward(self, pos):
        noise = torch.randn_like(pos) * self.noise_std.to(pos.device)
        return pos + noise

class ResidualDenoiseHead(nn.Module):
    """
    残差式几何重建分支：从 encoder 的特征空间预测 Δpos。
    输入为 encoder 输出特征 h_i，输出为 Δpos_i。
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h):
        return self.mlp(h)  # 输出 Δpos

class byol_denoise(nn.Module):
    """
    BYOL + 残差几何重建联合训练版本
    - online_encoder 输出结构特征
    - denoise_head 从特征预测 Δpos (重建 pos)
    - 同时进行 BYOL 对比 & 几何重建任务
    """
    def __init__(self, encoder, predictor,
                 hidden_dim=128, noise_std=0.02,
                 learnable_std=False, denoise_weight=1.0):
        super().__init__()
        # 1. online / target encoders
        self.online_encoder = encoder
        self.predictor = predictor
        self.target_encoder = copy.deepcopy(encoder)

        # 初始化 target encoder
        if hasattr(self.target_encoder, "reset_parameters"):
            self.target_encoder.reset_parameters()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 2. coordinate noise generator
        self.coord_noise = CoordNoise(noise_std, learnable_std)

        # 3. residual denoising head
        self.denoise_head = ResidualDenoiseHead(in_dim=128,
                                                hidden_dim=hidden_dim)

        # 4. loss 权重
        self.denoise_weight = denoise_weight

    def trainable_parameters(self):
        params = list(self.online_encoder.parameters()) + \
                 list(self.predictor.parameters()) + \
                 list(self.denoise_head.parameters())
        if isinstance(self.coord_noise.noise_std, nn.Parameter):
            params += [self.coord_noise.noise_std]
        return params

    @torch.no_grad()
    def update_target_network(self, mm):
        assert 0.0 <= mm <= 1.0, 'momentum needs to be between 0.0 and 1.0, got %.5f' % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x, return_denoise=False):

        # ===============================
        # 1️⃣ Online encoder 输入带噪坐标
        # ===============================
        if hasattr(online_x, "pos"):
            noisy_pos = self.coord_noise(online_x.pos)
            online_x = online_x.clone()
            online_x.pos = noisy_pos

        # ---- Encoder 前向传播 ----
        online_y, online_node = self.online_encoder(online_x)  # 返回全局特征与节点特征
        online_q = self.predictor(online_y)


        # ===============================
        # 2️⃣ Denoising 分支（预测 Δpos）
        # ===============================
        delta_pos = None
        if hasattr(online_x, "pos") and return_denoise:
            delta_pos = self.denoise_head(online_node)  # 输出 Δpos（残差）
            # 不在这里计算 loss，由外部 train_step 控制
        # ===============================
        # 3️⃣ Target encoder (no grad)
        # ===============================
        with torch.no_grad():
            target_y, target_node = self.target_encoder(target_x)

        # ===============================
        # 4️⃣ 返回
        # ===============================

        if return_denoise:
            return online_q, target_y, delta_pos
        else:
            return online_q, target_y



def get_features_from_encoder(model, loader, type):
    model.eval()  # 设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在的设备

    latents = []
    labels = []

    with torch.no_grad():  # 禁用梯度计算
        for data in loader:
            # 将数据转移到模型所在的设备
            data = data.to(device)
            # 获取特征表示
            latent, node = model(data)
            # 检查并处理 NaN 值
            if torch.isnan(latent).any():
                print("警告: 特征包含 NaN，进行清理")
                latent = torch.nan_to_num(latent, nan=0.0)
            # 添加到列表
            latents.append(latent.cpu())
            labels.append(getattr(data, type).cpu())

    # 合并所有批次的特征和标签
    all_latents = torch.cat(latents, dim=0)
    all_labels = torch.cat(labels, dim=0)

    # 再次检查是否有 NaN
    if torch.isnan(all_latents).any():
        print("严重警告: 合并后特征仍包含 NaN，进行最终清理")
        all_latents = torch.nan_to_num(all_latents, nan=0.0)

    return all_latents, all_labels


if __name__ == '__main__':

    config = {
        'name': 'new_BIL',
        'data': {
            'root': r'D:\PycharmProjects\B1\dataset\all',
            'path': r'D:\Dataset\Neuron',

            'act_root': r'D:\PycharmProjects\B1\dataset\act',
            'act_path': r'D:\Dataset\Neuron\act',

            'bil_root': r'D:\PycharmProjects\B1\dataset\bil',
            'bil_path': r'D:\Dataset\Neuron\bil',

            'keep_node': 512,
            'batch_size': 32,
            'num_workers': 4 if os.name != 'nt' else 0
        },
        'model': {
            'channels': 128,
            'num_layers': 3,
            'num_heads': 8,
            'pe_dim': 32,
            'proj_dim': 128,
            'pred_dim': 128,
            'dropout': 0.1,
            'attn_dropout': 0.1,
            'use_coord_loss': True,
            'coord_weight': 0.5,
            'ema_decay': 0.99
        },
        'training': {
            'epochs': 1000,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'eval_interval': 2,
            'log_dir': 'runs/byol_run',
            'save_dir': '../ckgs/ssl_act_layer',
            'lr_warmup_steps': 1000
        }
    }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using {device} for training.')
    transform_v2 = T.Compose([
        T.RandomRotate(90, axis=0),
        T.RandomRotate(90, axis=1),
        T.RandomRotate(90, axis=2),
        T.RandomFlip(axis=0),
        T.RandomFlip(axis=1),
        T.RandomFlip(axis=2),
        T.RandomJitter(0.01),
    ])
    transform_v3 = T.Compose([
        T.RandomRotate(90, axis=0),
        T.RandomRotate(90, axis=1),
        T.RandomRotate(90, axis=2),
        T.RandomFlip(axis=0),
        T.RandomFlip(axis=1),
        T.RandomFlip(axis=2),
        # T.RandomJitter(0.01),
    ])

    # 数据增强
    eval_transform = T.Compose([
        # NewBranchCutTransform(keep_nodes=256, protected_nodes=[0],
        #                       allow_disconnect=False, enable_branch_cut=False, max_branch=0),
        AddShortestPathPE(anchor_selection='kmeans', num_anchors=20, attr_name='pe'),
        AddHeatKernelPE_v2(time=[0.1, 1.0, 10], diag_attr_name='hk_pe', full_attr_name='bias'),
    ])


    pretrain_transform = MultiViewDataInjector([
        T.Compose([
            # RandomJitterNeurite(noise_std=0.01, translate=0.05, rotate=True, flip=True, seed=42),
            transform_v2,
            # NewBranchCutTransform(keep_nodes=128, protected_nodes=[0], allow_disconnect=False,
            #                       enable_branch_cut=True, max_branch=15),
            AddShortestPathPE(anchor_selection='kmeans', num_anchors=20, attr_name='pe'),
            AddHeatKernelPE_v2(time=[0.1, 1.0, 10], diag_attr_name='hk_pe', full_attr_name='bias'),
        ]),
        T.Compose([
            # RandomJitterNeurite(noise_std=0.01, translate=0.05, rotate=True, flip=True, seed=24),

            transform_v2,
            # NewBranchCutTransform(keep_nodes=128, protected_nodes=[0], allow_disconnect=False,
            #                       enable_branch_cut=True, max_branch=15),
            AddShortestPathPE(anchor_selection='kmeans', num_anchors=20, attr_name='pe'),
            AddHeatKernelPE_v2(time=[0.1, 1.0, 10], diag_attr_name='hk_pe', full_attr_name='bias'),
        ])
    ])

    # Dataset（去掉多余逗号）
    all_data = ACT(
        root=config['data']['act_root'],
        path=config['data']['act_path'],
        type='layer',
        data_name='act',
        keep_node=512,
        transform=pretrain_transform
    )
    # print(all_data[0])
    #
    # all_data = NeuronDataset(
    #     root=config['data']['act_root'],
    #     path=config['data']['act_path'],
    #     data_name='n7',
    #     keep_node=512,
    #     transform=pretrain_transform
    # )

    # act_evel_data = NeuronDataset(
    #     root=config['data']['act_root'],
    #     path=config['data']['act_path'],
    #     data_name='n7',
    #     keep_node=512,
    #     transform=eval_transform
    # )

    act_evel_data = ACT(
        root=config['data']['act_root'],
        path=config['data']['act_path'],
        type='layer',
        data_name='act',
        keep_node=512,
        transform=eval_transform
    )
    #
    # all_data = ACT(
    #     root=config['data']['act_root'],
    #     path=config['data']['act_path'],
    #     type='all',
    #     data_name='act',
    #     keep_node=512,
    #     transform=pretrain_transform
    # )
    # #
    # act_evel_data = ACT(
    #     root=config['data']['act_root'],
    #     path=config['data']['act_path'],
    #     type='layer',
    #     data_name='act',
    #     keep_node=512,
    #     transform=eval_transform
    # )

    # Train/validation splits
    act_labels = [data.layer_y.item() for data in act_evel_data]
    act_train_indices, act_val_indices = train_test_split(
        range(len(act_evel_data)),
        test_size=0.2,
        stratify=act_labels,
        random_state=12
    )
    act_train_dataset = act_evel_data.index_select(act_train_indices)
    act_val_dataset = act_evel_data.index_select(act_val_indices)

    # act_type_labels = [data.layer_y.item() for data in act_type_data]
    # act_type_train_indices, act_type_val_indices = train_test_split(
    #     range(len(act_type_data)), test_size=0.2, stratify=act_type_labels, random_state=12
    # )
    # act_type_train_dataset = act_type_data.index_select(act_type_train_indices)
    # act_type_val_dataset = act_type_data.index_select(act_type_val_indices)
    #
    #
    # bil_labels = [data.type_y.item() for data in bil_evel_data]
    # bil_train_indices, bil_val_indices = train_test_split(
    #     range(len(bil_evel_data)), test_size=0.2, stratify=bil_labels, random_state=12
    # )
    # bil_train_dataset = bil_evel_data.index_select(bil_train_indices)
    # bil_val_dataset = bil_evel_data.index_select(bil_val_indices)

    # DataLoaders
    all_loader = DataLoader(all_data, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'], drop_last=True)
    act_train_loader = DataLoader(act_train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'])
    act_val_loader = DataLoader(act_val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    # act_type_train_loader = DataLoader(act_type_train_dataset, batch_size=config['data']['batch_size'],
    #                           shuffle=True, num_workers=config['data']['num_workers'])
    # act_type_val_loader = DataLoader(act_type_val_dataset, batch_size=config['data']['batch_size'],
    #                         shuffle=False, num_workers=config['data']['num_workers'])

    # bil_train_loader = DataLoader(bil_train_dataset, batch_size=config['data']['batch_size'],
    #                           shuffle=True, num_workers=config['data']['num_workers'])
    # bil_val_loader = DataLoader(bil_val_dataset, batch_size=config['data']['batch_size'],
    #                         shuffle=False, num_workers=config['data']['num_workers'])

    print(f"Pretrain samples: {len(all_data)}")
    print(f"ACT train/val: {len(act_train_dataset)}/{len(act_val_dataset)}")
    # print(f'ACT type: {len(act_type_train_dataset)}/{len(act_type_val_dataset)}')
    # print(f"BIL train/val: {len(bil_train_dataset)}/{len(bil_val_dataset)}")


    # 构建模型
    Encoder = Transformer_encoder(
        in_features=3,
        channels=config['model']['channels'],
        pe_dim=config['model']['pe_dim'],
        num_layer=config['model']['num_layers'],
        heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        attn_dropout=config['model']['attn_dropout']
    )

    Predictor = MLP_Predictor(
        input_size=config['model']['channels'],
        output_size=config['model']['channels'],
        hidden_size=config['model']['channels']
    )

    model = byol_denoise(Encoder, Predictor).to(device)

    path = '../ckgs/ablation/n7/ssl_n7_0.7348.pt'
    state_dict = torch.load(path, map_location=device)
    model.online_encoder.load_state_dict(state_dict)
    #

    # Optimizer & Scheduler
    total_steps = config['training']['epochs'] * len(all_loader)
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    lr_scheduler = CosineDecayScheduler(config['training']['lr'], config['training']['lr_warmup_steps'],total_steps)
    mm_scheduler = CosineDecayScheduler(1 - config['model']['ema_decay'], 0, total_steps)
    def train_step(data, step):
        # MultiViewDataInjector -> (view1, view2)
        x1, x2 = data[0].to(device), data[1].to(device)
        # ---- 学习率 & 动量调度 ----
        lr = lr_scheduler.get(step)
        mm = 1 - mm_scheduler.get(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()

        # ---- 模型前向传播（包含去噪） ----
        q1, y2, delta1 = model(x1, x2, return_denoise=True)
        q2, y1, delta2 = model(x2, x1, return_denoise=True)

        # ---- BYOL 对比损失 ----
        loss_ssl = (1 - F.cosine_similarity(q1, y2.detach(), dim=-1).mean()
                    + 1 - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()) / 2

        gt_delta1 = x1.pos - x2.pos  # 或使用 (x1.pos_noisy - x1.pos) 视情况定义
        gt_delta2 = x2.pos - x1.pos
        loss_denoise = (F.mse_loss(delta1, gt_delta1) + F.mse_loss(delta2, gt_delta2)) / 2

        # ---- 总损失 ----
        loss = loss_ssl + 0.2 * loss_denoise
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)

        return loss


    def eval_knn(train_loader, val_loader, type):
        # 获取训练集特征和标签
        x_train, y_train = get_features_from_encoder(model.online_encoder, train_loader, type=type)

        # 获取验证集特征和标签
        x_val, y_val = get_features_from_encoder(model.online_encoder, val_loader, type=type)

        # 转换为 NumPy 数组
        x_train_np = x_train.numpy()
        y_train_np = y_train.numpy()
        x_val_np = x_val.numpy()
        y_val_np = y_val.numpy()

        # 最终检查 NaN
        if np.isnan(x_train_np).any() or np.isnan(x_val_np).any():
            print("最终警告: 特征中仍有 NaN 值，进行最终清理")
            x_train_np = np.nan_to_num(x_train_np, nan=0.0)
            x_val_np = np.nan_to_num(x_val_np, nan=0.0)

        # KNN分类
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(x_train_np, y_train_np)
        train_acc = neigh.score(x_train_np, y_train_np)
        val_acc = neigh.score(x_val_np, y_val_np)

        return train_acc, val_acc

    # Training loop
    min_loss = float('inf')
    act_max_train_knn = act_max_val_knn = act_type_max_val_knn = bil_max_val_knn = 0.0
    act_best_epoch = act_type_best_epoch = bil_best_epoch = 0
    step = 0

    for epoch in range(config['training']['epochs']):
        model.train()  # 设置为训练模式
        epoch_losses = []
        progress_bar = tqdm(all_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")

        for data in progress_bar:
            step += 1
            loss = train_step(data, step)
            epoch_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss)
            # 计算平均损失
        avg_loss = np.mean(epoch_losses)

        # Evaluation
        if epoch % config['training']['eval_interval'] == 0:
            model.eval()
            with torch.no_grad():
                act_train_knn, act_val_knn = eval_knn(act_train_loader, act_val_loader, type='layer_y')


            # if act_train_knn >= act_max_train_knn:
            #     act_max_train_knn = act_train_knn
            #     torch.save(model.online_encoder.state_dict(),
            #                os.path.join(r'../ckgs/ablation/bbp', f'ssl_bbp_train_{act_train_knn:.4f}.pt'))
            # Save best models
            if act_val_knn >= act_max_val_knn:
                act_max_val_knn = act_val_knn
                act_best_epoch = epoch
                torch.save(model.online_encoder.state_dict(),
                           os.path.join(r'../ckgs/ablation/act_layer', f'ssl_act_layer_{act_val_knn:.4f}.pt'))

            print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
            print(f"ACT | Train KNN: {act_train_knn:.4f}  Val KNN: {act_val_knn:.4f} (Best: {act_max_val_knn:.4f} @ epoch {act_best_epoch})")

        else:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        # Final save
        torch.save(model.online_encoder.state_dict(),
                       os.path.join(r'../ckgs/ablation/act_layer', 'act_layer_final_encoder.pt'))
    print(f"\nTraining completed!")
    print(f"Best ACT KNN: {act_max_val_knn:.4f} at epoch {act_best_epoch}")
    print(f"Best BIL KNN: {bil_max_val_knn:.4f} at epoch {bil_best_epoch}")