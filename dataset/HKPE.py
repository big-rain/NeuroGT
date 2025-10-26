#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/19 15:54
# @Author  : ShengPengpeng
# @File    : PE.py
# @Description :


from torch.nn.functional import normalize
from torch_geometric.transforms import Compose

import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from typing import Union, List, Optional
from engine.functions import add_node_attr





class AddHeatKernelPE_v2(BaseTransform):
    """
    Optimized Heat Kernel Positional Encoding using PyTorch's native eigh.

    Args:
        time (float or list): Heat diffusion time(s)
        use_normalized_laplacian (bool): Whether to use normalized Laplacian
        diag_attr_name (str): Attribute name for diagonal values (HKPE_diag)
        full_attr_name (str): Attribute name for full heat kernel matrix
        cache (bool): Cache result for reuse
    """
    def __init__(self,
                 time: Union[float, List[float]] = 1.0,
                 use_normalized_laplacian: bool = True,
                 diag_attr_name: Optional[str] = 'HKPE_diag',
                 full_attr_name: Optional[str] = 'heat_kernel_full',
                 device=torch.device('cuda:0'),
                 cache: bool = False):
        self.time = [time] if isinstance(time, float) else time
        self.use_normalized_laplacian = use_normalized_laplacian
        self.diag_attr = diag_attr_name
        self.full_attr = full_attr_name
        self.cache = cache
        self._cached = None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, data: Data) -> Data:

        data = data.to(self.device)
        if self.cache and self._cached is not None:
            diag_pe, full_pe = self._cached
        else:
            num_nodes = data.num_nodes
            edge_index = data.edge_index
            edge_weight = getattr(data, 'edge_attr', None)
            edge_weight = edge_weight.squeeze(-1)


            # Step 1: Get Laplacian matrix (sparse COO format)
            normalization = 'sym' if self.use_normalized_laplacian else 'none'
            edge_index_lap, edge_weight_lap = get_laplacian(
                edge_index, edge_weight, normalization=normalization, num_nodes=num_nodes)

            # Step 2: Convert to dense PyTorch tensor
            L = to_dense_adj(edge_index_lap, edge_attr=edge_weight_lap,
                            max_num_nodes=num_nodes).squeeze(0)  # [N, N]

            # Step 3: Eigen decomposition with torch.linalg.eigh
            evals, evects = torch.linalg.eigh(L)  # Automatically uses GPU if available

            # Step 4: Filter and normalize eigenvectors
            mask = evals > 1e-8  # Filter near-zero eigenvalues
            evals, evects = evals[mask], evects[:, mask]
            evects = F.normalize(evects, p=2, dim=0)  # L2 normalize

            # Step 5: Compute Heat Kernel for each time
            diag_list, full_list = [], []
            for t in self.time:
                D = torch.diag(torch.exp(-t * evals))
                heat_kernel = evects @ D @ evects.T  # [N, N]
                diag_list.append(heat_kernel.diag().unsqueeze(1))  # [N, 1]
                if self.full_attr:
                    full_list.append(heat_kernel.unsqueeze(0))  # [1, N, N]

            diag_pe = torch.cat(diag_list, dim=1)  # [N, T]

            # Only compute full_pe if enabled
            if self.full_attr and full_list:
                full_pe = torch.cat(full_list, dim=0).mean(dim=0)  # [N, N]
            else:
                full_pe = None

        # Step 6: Store results
        if self.diag_attr:
            data[self.diag_attr] = diag_pe
        if self.full_attr and full_pe is not None:
            data[self.full_attr] = full_pe

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(time={self.time}, '
                f'normalized={self.use_normalized_laplacian}, '
                f'diag_attr={self.diag_attr}, '
                f'full_attr={self.full_attr})')




class AddHeatKernelPE_v1(BaseTransform):
    """
    将基于热核的位置编码应用于输入图数据。

    参数：

        dim (int, 可选)：位置编码的维度。 (默认值：10)

        time (float 或 list, 可选)：热扩散的时间参数。如果提供列表，将连接每个时间点的编码。(默认值：1.0)

        use_normalized_laplacian (bool, 可选)：是否使用归一化拉普拉斯矩阵。(默认值：True)

        attr_name (str, 可选)：存储位置编码的属性名称。如果为 None，则使用 add_node_attr 的默认行为。(默认值：None)

        cache (bool, 可选)：如果为 True，将缓存计算出的编码。(默认值：False)

    """

    def __init__(self,
                 time: Union[float, List[float]] = 1.0,
                 use_normalized_laplacian: bool = True,
                 attr_name: Optional[str] = 'HKPE',
                 cache: bool = False):

        self.time = [time] if isinstance(time, float) else time
        self.use_normalized_laplacian = use_normalized_laplacian
        self.attr_name = attr_name
        self.cache = cache
        self._cached_pe = None

    def forward(self, data: Data) -> Data:
        if self._cached_pe is not None:
            return add_node_attr(data, self._cached_pe, self.attr_name)

        # Convert edge_index to dense adjacency matrix
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        # 取 edge_weight 或默认 1
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else torch.ones(edge_index.size(1))

        adj = torch.zeros((num_nodes, num_nodes),
                         dtype=torch.float,
                         device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = edge_weight

        # 构造拉普拉斯矩阵
        deg = adj.sum(dim=1)
        if self.use_normalized_laplacian:
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            L = torch.eye(num_nodes, device=adj.device) - (
                deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1))
        else:
            L = torch.diag(deg) - adj

        # 拉普拉斯谱分解
        evals, evects = torch.linalg.eigh(L)
        mask = evals > 1e-8
        evals, evects = evals[mask], evects[:, mask]
        evects = normalize(evects, p=2, dim=0)

        # 计算热核扩散矩阵 & HKPE
        pe_list = []
        for t in self.time:
            heat_kernel = evects @ torch.diag(torch.exp(-t * evals)) @ evects.T
            pe_list.append(heat_kernel.sum(dim=1, keepdim=True))  # 每个节点的热扩散能量

        pe = torch.cat(pe_list, dim=1) if len(pe_list) > 1 else pe_list[0]

        # add_node_attr(data, heat_kernel, 'heat_mm')
        add_node_attr(data, pe, self.attr_name)
        return add_node_attr(data, heat_kernel, 'heat_mm')


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}, '
                f'time={self.time}, '
                f'normalized={self.use_normalized_laplacian}, '
                f'attr_name={self.attr_name})')



class AddHeatKernelPE_v3(BaseTransform):

    r"""
    Node-aware Regularized Heat Kernel Positional Encoding:
        H_γ(t) = exp(-t * (L + Γ))
    where Γ = diag(γ_i), γ_i = γ0 / r_i

    Args:
        time (float or list): Diffusion time(s)
        gamma0 (float): Base decay factor γ₀
        use_normalized_laplacian (bool): Whether to use normalized Laplacian
        radius_attr (str): Node attribute used for r_i (default: 'degree')
        diag_attr_name (str): Name for diagonal HKPE output
        full_attr_name (str): Name for full heat kernel matrix
        device (torch.device): Computation device
        cache (bool): Cache result for reuse
    """
    def __init__(self,
                 time=1.0,
                 gamma0=0.1,
                 use_normalized_laplacian=True,
                 radius_attr='x',
                 diag_attr_name='HKPE_diag',
                 full_attr_name='heat_kernel_full',
                 device=None,
                 cache=False):
        self.time = [time] if isinstance(time, (float, int)) else time
        self.gamma0 = gamma0
        self.use_normalized_laplacian = use_normalized_laplacian
        self.radius_attr = radius_attr
        self.diag_attr = diag_attr_name
        self.full_attr = full_attr_name
        self.cache = cache
        self._cached = None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data: Data) -> Data:
        data = data.to(self.device)

        if self.cache and self._cached is not None:
            diag_pe, full_pe = self._cached
        else:
            num_nodes = data.num_nodes
            edge_index = data.edge_index
            edge_weight = getattr(data, 'edge_attr', None)
            if edge_weight is not None and edge_weight.dim() > 1:
                edge_weight = edge_weight.squeeze(-1)

            # Step 1: Build Laplacian
            normalization = 'sym' if self.use_normalized_laplacian else 'none'
            edge_index_lap, edge_weight_lap = get_laplacian(
                edge_index, edge_weight, normalization=normalization, num_nodes=num_nodes)
            L = to_dense_adj(edge_index_lap, edge_attr=edge_weight_lap,
                             max_num_nodes=num_nodes).squeeze(0).to(self.device)

            # Step 2: Compute radius r_i
            if hasattr(data, self.radius_attr):
                features = getattr(data, self.radius_attr).to(self.device)
                r = features[:, 1]
            else:
                # Default: use node degree
                deg = torch.zeros(num_nodes, device=self.device)
                deg.index_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float, device=self.device))
                r = deg + 1e-6  # avoid zero
            gamma = self.gamma0 * r

            # Step 3: Regularized Laplacian L_reg = L + diag(gamma)
            L_reg = L + torch.diag(gamma)

            # Step 4: Eigendecomposition
            evals, evects = torch.linalg.eigh(L_reg)
            mask = evals > 1e-8
            evals, evects = evals[mask], evects[:, mask]
            evects = F.normalize(evects, p=2, dim=0)

            # Step 5: Heat Kernel computation
            diag_list, full_list = [], []
            for t in self.time:
                exp_term = torch.exp(-t * evals)
                H = evects @ (exp_term.unsqueeze(0) * evects.T)
                diag_list.append(H.diag().unsqueeze(1))
                if self.full_attr:
                    full_list.append(H.unsqueeze(0))

            diag_pe = torch.cat(diag_list, dim=1)  # [N, T]
            full_pe = torch.cat(full_list, dim=0).mean(dim=0) if full_list else None

            if self.cache:
                self._cached = (diag_pe, full_pe)

        if self.diag_attr:
            data[self.diag_attr] = diag_pe
        if self.full_attr and full_pe is not None:
            data[self.full_attr] = full_pe

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(time={self.time}, gamma0={self.gamma0}, '
                f'normalized={self.use_normalized_laplacian}, radius_attr={self.radius_attr})')





if __name__ == '__main__':
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.datasets import TUDataset
    import torch
    from torch_geometric.utils import to_networkx
    import networkx as nx
    from time import time
    from dataloader.ACT import ACT    # 两个小图
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'D:/Dataset/Neuron/'
    act_root = '../dataset/bil'

    transform = Compose([
        AddHeatKernelPE_v2( time=[0.5, 1.0, 10], diag_attr_name='heat_pe', full_attr_name='full_pe'),
        # AddHeatKernelPE_v1(time=[1.0, 5.0, 10.0], attr_name='heat_1'),
        AddHeatKernelPE_v3(time=[0.5, 1.0, 10.0], diag_attr_name='heat_2'),
        # AddRandomWalkPE(walk_length=20, attr_name='random_walk'),
        # AddLaplacianEigenvectorPE(k=2, attr_name='laplacian_eigenvector', is_undirected=True),
    ])


    act_dataset = ACT(
        root=act_root,
        path=path,
        type='layer',
        data_name='bil',
        keep_node=512,
        transform=transform,
    )
    data = act_dataset[0]
    torch.set_printoptions(threshold=float('inf'))
    print(act_dataset[0].heat_2)
    print(act_dataset[1].heat_pe)
    # print(act_dataset[0].heat_1)
    # print(act_dataset[0].full_pe)
    # print(act_dataset[0].heat_mm)
    # print(act_dataset[0].random_walk)


