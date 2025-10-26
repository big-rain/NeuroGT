"""Path Encoder"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    def __init__(self, hidden_dim=64, type_vocab_size=8, strahler_vocab_size=10):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ====================
        # 离散特征嵌入
        # ====================
        self.type_embedding = self._create_embedding(type_vocab_size, hidden_dim, is_binary=False)
        self.strahler_embedding = self._create_embedding(strahler_vocab_size, hidden_dim, is_binary=False)
        self.children_embedding = self._create_embedding(128, hidden_dim, is_binary=False)
        self.terminal_embed = self._create_embedding(2, hidden_dim, is_binary=True)
        self.branch_embed = self._create_embedding(2, hidden_dim, is_binary=True)

        # ====================
        # 连续特征处理
        # ====================
        self.scalar_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # ====================
        # 特征融合      x[:, 1],  # radius
        #             x[:, 2],  # distance_from_root
        #             x[:, 3],  # path_length_to_terminal
        #             x[:, 7],  # subtree-size
        # ====================
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # 自适应特征权重
        self.feature_weights = nn.Parameter(torch.ones(6))

        # 初始化
        self._init_weights()

    def _create_embedding(self, num_embeddings, embedding_dim, is_binary=False):
        """创建并初始化嵌入层"""
        emb = nn.Embedding(num_embeddings, embedding_dim)
        if is_binary:
            nn.init.constant_(emb.weight, 0.0)  # 二值特征初始化为零
        else:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        return emb

    def _init_weights(self):
        """初始化投影层权重"""
        # 连续特征投影初始化
        for layer in self.scalar_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)

        # 特征融合层初始化
        for layer in self.fusion_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # 解包特征
        # x = data.x

        # ====================
        # 离散特征嵌入
        # ====================
        type_feat = self.type_embedding(x[:, 0].long())
        strahler_feat = self.strahler_embedding(x[:, 8].long())
        children_feat = self.children_embedding(x[:, 6].long())
        is_branch_feat = self.branch_embed(x[:, 4].long())
        is_terminal_feat = self.terminal_embed(x[:, 5].long())

        # ====================
        # 连续特征处理
        # ====================
        scalar_input = torch.stack([
            x[:, 1],  # radius
            x[:, 2],  # distance_from_root
            x[:, 3],  # path_length_to_terminal
            x[:, 7],  # subtree-size
        ], dim=-1).float()  # 确保为浮点类型

        scalar_feat = self.scalar_proj(scalar_input)
        # ====================
        # 特征融合
        # ====================
        # 收集所有特征
        features = [
            type_feat,
            strahler_feat,
            children_feat,
            is_branch_feat,
            is_terminal_feat,
            scalar_feat
        ]

        # 应用自适应特征权重
        weights = F.softmax(self.feature_weights, dim=0)
        weighted_features = [w * feat for w, feat in zip(weights, features)]

        # 拼接加权特征
        node_feat = torch.cat(weighted_features, dim=-1)

        return self.fusion_proj(node_feat)

    def get_feature_weights(self):
        """获取特征权重（用于解释性分析）"""
        weights = F.softmax(self.feature_weights, dim=0)
        feature_names = ["type", "strahler", "children", "branch", "terminal", "scalar"]
        return {name: weight.item() for name, weight in zip(feature_names, weights)}



class HeatKernelEncoder(nn.Module):
    r"""
    热核扩散位置编码器：将节点对之间的扩散关系转换为可学习的嵌入向量

    参数：
    min_val：float
        扩散值归一化下限（建议0）
    max_val：float
        扩散值归一化上限（建议1）
    num_bins：int
        离散化桶的数量
    num_heads：int，可选
        注意力头数（默认为1）
    log_transform：bool，可选
        是否对输入值进行对数变换（默认为True）
    """
    def __init__(self, min_val, max_val, num_bins, num_heads=1, log_transform=True):
        super().__init__()
        self.min_val  = min_val
        self.max_val  = max_val
        self.num_bins  = num_bins
        self.num_heads  = num_heads
        self.log_transform  = log_transform

        # 离散化参数
        self.bin_edges = torch.linspace(min_val, max_val, num_bins + 1)

        # 嵌入表：额外位置0用于无效值
        self.embedding_table  = nn.Embedding(
            num_bins + 2, num_heads, padding_idx=0
        )

    def forward(self, diffusion_matrix, batch):
        """
        输入：
        diffusion_matrix：torch.Tensor
            节点对的扩散矩阵，形状：(B, N, N)
            - 无效位置应标记为负数（如-1）

        输出：
        encoding：torch.Tensor
            扩散关系编码，形状：(B, N, N, H)
        """

        # 获取 batch 大小和每图节点数
        B = batch.max().item() + 1
        total_nodes = diffusion_matrix.size(0)
        N = total_nodes // B
        # reshape 成 [B, N, N]
        diffusion_matrix = diffusion_matrix.view(B, N, N)
        original = diffusion_matrix.clone()

        # 对数变换（处理指数衰减特性）
        if self.log_transform:
            diffusion_matrix = torch.log(diffusion_matrix  + 1e-12)

        # 创建无效值掩码
        valid_mask = (diffusion_matrix >= 0)
        invalid_mask = ~valid_mask

        # 归一化处理
        normalized = torch.zeros_like(diffusion_matrix)
        normalized[valid_mask] = (diffusion_matrix[valid_mask] - self.min_val) / (self.max_val - self.min_val)

        # 离散化分桶
        bin_indices = torch.bucketize(
            normalized,
            self.bin_edges.to(diffusion_matrix.device),
            right=True
        ).long()

        # 处理无效位置：索引0（padding_idx）
        bin_indices[invalid_mask] = 0

        # 偏移索引（为有效值预留索引1~num_bins+1）
        bin_indices += 1
        # 确保索引有效
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins + 1)

        # 嵌入表查询
        diffusion_encoding = self.embedding_table(bin_indices)

        # 添加距离敏感偏置（增强位置感知）
        distance_bias = 1 / (original + 1e-6)  # 防止除零
        diffusion_encoding = diffusion_encoding * distance_bias.unsqueeze(-1)
        # 调整形状为 [B, H, N, N]（用于多头 attention bias）
        diffusion_encoding = diffusion_encoding.permute(0, 3, 1, 2)
        return diffusion_encoding



class PathEncoder(nn.Module):
    """
    路径编码器，如在《Transformer真的不擅长图表示吗？》中介绍的模块，是一个可学习的路径嵌入模块，它将每对节点之间的最短路径作为注意力偏置进行编码。

    参数
    ----------
    max_len : int
        每条路径中要编码的最大边数。超出部分的路径将被截断，即截断序列号大于或等于：attr:`max_len`的边。
    feat_dim : int
        输入图中边特征的维度。
    num_heads : int, 可选
        如果应用多头注意力机制，则为注意力头的数量。默认值：1。
    """
    def __init__(self, max_len, feat_dim, num_heads=1):
        super().__init__()
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(max_len * num_heads, feat_dim)

    def forward(self, dist, path_data):

        shortest_distance = torch.clamp(dist, min=1, max=self.max_len)
        edge_embedding = self.embedding_table.weight.reshape(
            self.max_len, self.num_heads, -1
        )
        path_encoding = torch.div(
            torch.einsum("bxyld,lhd->bxyh", path_data, edge_embedding).permute(
                3, 0, 1, 2
            ),
            shortest_distance,
        ).permute(1, 2, 3, 0)
        return path_encoding


class SpatialEncoder(nn.Module):
    r"""
    这个模块是一个可学习的空间嵌入模块，它编码了每对节点之间的最短距离作为注意力偏置。

    max_dist: int
        最短路径距离的上限，用于编码每对节点之间的距离。
        所有距离将被限制在 [0, max_dist] 范围内。
    num_heads: int, 可选
        如果应用了多头注意力机制，则为注意力头的数量。
        默认值：1。
    """

    def __init__(self, max_dist, num_heads=1):
        super(SpatialEncoder, self).__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads

        # deactivate node pair between which the distance is -1
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, padding_idx=0
        )
    def forward(self, dist):
        """
        :parameter:  dist: torch.Tensor

            批处理图的最短路径距离，带有-1填充，形状为：math:`(B, N, N)` 的张量，其中：math:`B` 是批处理图的批量大小，：math:`N` 是节点的最大数量。

        :return: torch.Tensor

            返回注意力偏差作为空间编码，形状为：math:`(B, N, N, H)`，其中：math:`H` 是：attr:`num_heads`。
        """
        spatial_encoding = self.embedding_table(
            torch.clamp(
                dist,
                min=-1,
                max=self.max_dist,
            )
            + 1
        )
        return spatial_encoding



class DegreeEncoder(nn.Module):
    r"""
    这个模块是一个可学习的度嵌入模块。

    max_degree : int
        要编码的最大度数。
        每个度数将被限制在[0, ``max_degree``]范围内。
    embedding_dim : int
        嵌入向量的输出维度。
    direction : str, 可选
        要编码的方向，从 ``in``、``out`` 和 ``both`` 中选择。
        ``both`` 会同时编码两个方向的度数，并输出它们的和。
        默认值：``both``。
    Example
    -------
    >>> import torch as th
    >>> from torch.nn.utils.rnn import pad_sequence

    >>> g1 = dgl.graph(([0,0,0,1,1,2,3,3], [1,2,3,0,3,0,0,1]))
    >>> g2 = dgl.graph(([0,1], [1,0]))
    >>> in_degree = pad_sequence([g1.in_degrees(), g2.in_degrees()], batch_first=True)
    >>> out_degree = pad_sequence([g1.out_degrees(), g2.out_degrees()], batch_first=True)
    >>> print(in_degree.shape)
    torch.Size([2, 4])
    >>> degree_encoder = DegreeEncoder(5, 16)
    >>> degree_embedding = degree_encoder(th.stack((in_degree, out_degree)))
    >>> print(degree_embedding.shape)
    torch.Size([2, 4, 16])
    """

    def __init__(self, max_degree, embedding_dim, direction='both'):
        super(DegreeEncoder, self).__init__()
        self.max_degree = max_degree

        if direction == 'both':
            self.encoder1 = nn.Embedding(max_degree + 1, embedding_dim, padding_idx=0)
            self.encoder2 = nn.Embedding(max_degree + 1, embedding_dim, padding_idx=0)
        else:
            self.encoder = nn.Embedding(max_degree + 1, embedding_dim, padding_idx=0)
        self.max_degree = max_degree

    def forward(self, degrees):
        """
        :param degree:  Tensor

            如果 :attr:`方向` 是 ``both``，它应该在批次图的入度和出度上堆叠，并用零填充，形状为 :math:`(2, B, N)` 的张量。

            否则，它应该在批次图的入度或出度上用零填充，形状为 :math:`(B, N)` 的张量，其中 :math:`B` 是批次图的批次大小，:math:`N` 是节点的最大数量。

        :return: Tensor

            返回形状为 :math:`(B, N, d)` 的度嵌入向量，其中 :math:`d` 是 :attr:`embedding_dim`。
        """
        degrees = torch.clamp(degrees, min=0, max=self.max_degree)

        if self.direction == "in":
            assert len(degrees.shape) == 2
            degree_embedding = self.encoder(degrees)
        elif self.direction == "out":
            assert len(degrees.shape) == 2
            degree_embedding = self.encoder(degrees)
        elif self.direction == "both":
            assert len(degrees.shape) == 3 and degrees.shape[0] == 2
            degree_embedding = self.encoder1(degrees[0]) + self.encoder2(
                degrees[1]
            )
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", '
                f"but got {self.direction}"
            )
        return degree_embedding
