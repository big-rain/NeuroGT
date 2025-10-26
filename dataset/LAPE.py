import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian


class RobustLaplacianEigenPE(BaseTransform):
    """
    Robust Laplacian Eigenvector Positional Encoding with better error handling.
    """

    def __init__(self, k: int, attr_name: str = "laplacian_eigenvector_pe",
                 normalization: str = "sym", use_dense: bool = False):
        self.k = k
        self.attr_name = attr_name
        self.normalization = normalization
        self.use_dense = use_dense

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        if num_nodes is None:
            num_nodes = data.edge_index.max().item() + 1

        # 使用CPU进行计算（更稳定）
        device = 'cpu'

        # 移到CPU并确保float32
        edge_index = data.edge_index.to(device)
        if data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device).float()
        else:
            edge_weight = None

        # 计算拉普拉斯矩阵
        edge_index, edge_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization=self.normalization,
            num_nodes=num_nodes,
        )

        # 构造拉普拉斯矩阵 - 使用稠密矩阵确保稳定性
        from torch_geometric.utils import to_dense_adj
        try:
            # 尝试构造稠密矩阵
            L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
            L = L.float()

            # 计算特征分解
            evals, evecs = torch.linalg.eigh(L)

            # 排序特征值
            evals, indices = torch.sort(evals)
            evecs = evecs[:, indices]

            # 选择特征向量
            start_idx = 1 if evals[0].abs() < 1e-6 else 0
            end_idx = min(start_idx + self.k, evecs.size(1))

            pe = evecs[:, start_idx:end_idx]

            # 如果特征向量不足，填充零
            if pe.size(1) < self.k:
                pad_size = self.k - pe.size(1)
                padding = torch.zeros(num_nodes, pad_size, dtype=pe.dtype)
                pe = torch.cat([pe, padding], dim=1)

            # 随机符号翻转
            sign = torch.randint(0, 2, (pe.size(1),), dtype=torch.float32) * 2 - 1
            pe = pe * sign.unsqueeze(0)

            # 归一化
            pe = F.normalize(pe, p=2, dim=0)

            # 存储结果
            data[self.attr_name] = pe

        except Exception as e:
            print(f"Error computing Laplacian PE: {e}")
            # 失败时返回零向量
            data[self.attr_name] = torch.zeros(num_nodes, self.k)

        return data