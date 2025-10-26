import torch
from torch import nn
from torch_geometric.data import Batch

import numpy as np
import torch
import networkx as nx
from typing import List, Optional, Literal
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

class ShortestPathBias(nn.Module):
    def __init__(self,
                 num_heads: int = 8,
                 scale: float = 10.0,
                 is_directed: bool = True,
                 edge_weight_name: str = 'edge_weight',
                 inf_replace: float = 1.0,
                 return_shape: str = 'BH',  # 'BH' or 'B,H'
                 ):
        """
        基于最短路径的注意力偏置计算模块。

        参数:
            num_heads (int): 注意力头数。
            scale (float): 缩放因子，最终 bias = -scale * norm_dist。
            is_directed (bool): 是否是有向图。
            edge_weight_name (str): Batch 中边权重的字段名。
            inf_replace (float): 不可达路径的替代距离。
            return_shape (str): 返回格式，'BH' 输出 [B*H, N, N]，'B,H' 输出 [B, H, N, N]。
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.is_directed = is_directed
        self.edge_weight_name = edge_weight_name
        self.inf_replace = inf_replace
        assert return_shape in ['BH', 'B,H']
        self.return_shape = return_shape

    def forward(self, batch_data: Batch):
        device = batch_data.x.device
        batch_data = batch_data.to(device)

        edge_index = batch_data.edge_index
        edge_weight = getattr(batch_data, self.edge_weight_name, None)
        batch = batch_data.batch
        num_graphs = batch_data.num_graphs
        max_num_nodes = batch.bincount().max().item()

        # 初始化输出 [B, N_max, N_max]
        attn_bias = torch.full((num_graphs, max_num_nodes, max_num_nodes),
                               float('inf'), device=device)

        for gid in range(num_graphs):
            node_mask = (batch == gid)
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            sub_edge_index = edge_index[:, edge_mask]
            n = node_mask.sum().item()

            # 映射节点索引
            old_nodes = torch.where(node_mask)[0]
            node_map = {old.item(): new for new, old in enumerate(old_nodes)}

            # 初始化距离矩阵
            dist = torch.full((n, n), float('inf'), device=device)
            dist.fill_diagonal_(0)

            # 处理边权
            weights = edge_weight[edge_mask] if edge_weight is not None else torch.ones(sub_edge_index.size(1), device=device)

            for (u, v), w in zip(sub_edge_index.T, weights):
                if u.item() == v.item():
                    continue  # 跳过自环

                u_mapped = node_map[u.item()]
                v_mapped = node_map[v.item()]
                dist[u_mapped, v_mapped] = w
                if not self.is_directed:
                    dist[v_mapped, u_mapped] = w

            # Floyd-Warshall
            for k in range(n):
                dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

            # 替换 inf
            finite = dist[dist != float('inf')]
            unreachable = finite.max().item() * 2 if len(finite) > 0 else self.inf_replace
            dist[dist == float('inf')] = unreachable

            # 填入结果矩阵
            attn_bias[gid, :n, :n] = dist

        # 缩放归一化
        max_val = attn_bias.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        attn_bias = -self.scale * (attn_bias / max_val)  # [B, N, N]
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=-1e6, neginf=-1e6)
        # 扩展多头
        attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, N, N]
        if self.return_shape == 'BH':
            attn_bias = attn_bias.flatten(0, 1)  # [B*H, N, N]
        return attn_bias





def add_node_attr(data: Data, attr: torch.Tensor, attr_name: str = "SPE"):
    setattr(data, attr_name, attr)
    return data

class AddShortestPathPE(BaseTransform):
    def __init__(self,
                 num_anchors: int = 10,
                 anchor_selection: Literal[
                     'random', 'degree', 'pagerank',
                     'closeness', 'betweenness', 'kmeans',
                     'eigenvector'
                 ] = 'random',
                 norm: bool = True,
                 attr_name: str = "SPE",
                 custom_anchors: Optional[List[int]] = None,
                 cache_path: Optional[str] = None):
        self.num_anchors = num_anchors
        self.anchor_selection = anchor_selection
        self.norm = norm
        self.attr_name = attr_name
        self.custom_anchors = custom_anchors
        self.cache_path = cache_path

    def forward(self, data: Data) -> Data:
        G = self._build_graph(data)
        nodes = list(G.nodes)
        anchors = self._get_anchors(data, G, nodes)
        spe_matrix = self._compute_shortest_paths(data, G, nodes, anchors)

        if self.norm:
            scaler = RobustScaler(quantile_range=(5, 95))
            spe_matrix = scaler.fit_transform(spe_matrix)

        data = add_node_attr(data, torch.tensor(spe_matrix, dtype=torch.float), attr_name=self.attr_name)

        return data

    def _build_graph(self, data: Data) -> nx.Graph:
        G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy().T

        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weights = data.edge_weight.cpu().numpy()
            for (u, v), w in zip(edge_index, edge_weights):
                G.add_edge(u, v, weight=w)
        else:
            G.add_edges_from(edge_index)
            for u, v in G.edges():
                G[u][v]['weight'] = 1.0
        return G

    def _get_anchors(self, data: Data, G: nx.Graph, nodes: List[int]) -> List[int]:
        if self.custom_anchors is not None:
            self.num_anchors = len(self.custom_anchors)
            return self.custom_anchors

        if self.anchor_selection == 'kmeans':
            if hasattr(data, 'pos') and data.pos is not None:
                pos_array = data.pos.cpu().numpy()  # shape [num_nodes, pos_dim]
                kmeans = KMeans(n_clusters=self.num_anchors, n_init='auto', random_state=42).fit(pos_array)
                # 选每个 cluster 中离中心最近的点作为 anchor
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                anchors = []
                for i in range(self.num_anchors):
                    cluster_idx = np.where(labels == i)[0]
                    if len(cluster_idx) == 0:
                        continue
                    cluster_points = pos_array[cluster_idx]
                    dists = np.linalg.norm(cluster_points - centers[i], axis=1)
                    closest_idx = cluster_idx[np.argmin(dists)]
                    anchors.append(int(closest_idx))
                return anchors

        centrality_funcs = {
            'random': lambda G, n: np.random.choice(n, self.num_anchors, replace=False).tolist(),
            'degree': lambda G, n: self._topk_nodes(nx.degree_centrality(G), self.num_anchors),
            'pagerank': lambda G, n: self._topk_nodes(nx.pagerank(G), self.num_anchors),
            'closeness': lambda G, n: self._topk_nodes(nx.closeness_centrality(G), self.num_anchors),
            'betweenness': lambda G, n: self._topk_nodes(nx.betweenness_centrality(G), self.num_anchors),
            'eigenvector': lambda G, n: self._topk_nodes(nx.eigenvector_centrality(G, max_iter=1000), self.num_anchors),
        }

        return centrality_funcs.get(self.anchor_selection, centrality_funcs['random'])(G, nodes)

    def _topk_nodes(self, scores: dict, k: int) -> List[int]:
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:k]]

    def _compute_shortest_paths(self, data: Data, G: nx.Graph, nodes: List[int], anchors: List[int]) -> np.ndarray:
        try:
            n, m = len(nodes), len(anchors)
            dist_matrix = np.full((n, m), np.inf, dtype=np.float32)

            for i, anchor in enumerate(anchors):
                try:
                    lengths = nx.single_source_dijkstra_path_length(G, anchor, weight='weight')
                    for node, dist in lengths.items():
                        dist_matrix[node, i] = dist
                except nx.NetworkXNoPath:
                    continue
            max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)])
            unreachable = max_dist * 2 if max_dist > 0 else 100.0
            dist_matrix[np.isinf(dist_matrix)] = unreachable
        except Exception as e:
            print(f"计算出错时的图信息: {data}")
            print(f"- 节点数: {len(nodes)}")
            print(f"- 边数: {G.number_of_edges()}")
            print(f"- 锚点数: {len(anchors)}")
            raise
        return dist_matrix




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

    transform_v1 =  AddShortestPathPE(anchor_selection='kmeans', attr_name='cpu_spe')
    # transform_v2 = TensorShortestPathPE(anchor_selection='degree', attr_name='gpu_spe')

    act_dataset = ACT(
        root=act_root,
        path=path,
        type='layer',
        data_name='bil',
        keep_node=512,
        # transform=transform,
    )
    data = act_dataset[0].clone()
    data = data.to('cuda')  # 如果你用GPU
    print(data)
    torch.set_printoptions(threshold=float('inf'))

    start = time()
    data_aug_v1 = transform_v1(data.clone())
    end = time()
    print(end - start)
    print(data_aug_v1.cpu_spe)


