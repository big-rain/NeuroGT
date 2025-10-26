import torch

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from dataloader.loader import NeuronDataset
from torch_geometric.loader import DataLoader

from model.ssl_3 import byol_gg
# from model.ssl_3 import Transformer_encoder, MLP_Predictor, byol_gg
from model.ssl_4 import Transformer_encoder, MLP_Predictor, byol_denoise
from dataloader.ACT import ACT
import os
import torch_geometric.transforms as T
from dataloader.augment import NewBranchCutTransform
from dataloader.HKPE import AddHeatKernelPE_v2
from dataloader.SPE import AddShortestPathPE
from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
import numpy as np


@torch.no_grad()
def extract_features(model, loader, device):
    all_features = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        feats,node = model(data)  # 你的模型需要有 get_features 方法
        all_features.append(feats.cpu())
        all_labels.append(data.layer_y.cpu())
    return torch.cat(all_features), torch.cat(all_labels)


# ===== 3. 自动搜索最佳K =====
def eval_knn_auto_k(x_train, y_train, x_val, y_val, k_range=(1, 20)):
    k_values = range(k_range[0], k_range[1] + 1)
    val_scores = []

    for k in k_values:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)
        val_scores.append(neigh.score(x_val, y_val))
    print(val_scores)
    best_k = k_values[int(np.argmax(val_scores))]
    best_val_acc = max(val_scores)

    # 再用最佳K计算训练集准确率
    neigh_best = KNeighborsClassifier(n_neighbors=best_k)
    neigh_best.fit(x_train, y_train)
    train_acc = neigh_best.score(x_train, y_train)

    return train_acc, best_val_acc, best_k




def eval_knn_auto_avg_k(x_train, y_train, x_val, y_val, k_range=(1, 20)):
    """
    返回:
    train_acc: 最佳K值下的训练集准确率
    best_val_acc: 最佳K值下的验证集准确率
    best_val_avg_precision: 最佳K值下的验证集平均类别精度
    best_k: 最佳K值
    val_scores: 所有K值的验证集准确率列表
    """
    k_values = range(k_range[0], k_range[1] + 1)
    val_acc_scores = []
    val_precision_scores = []

    # 预先计算所有预测结果以提高效率
    for k in k_values:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)

        # 验证集预测
        y_val_pred = neigh.predict(x_val)

        # 计算准确率
        val_acc = accuracy_score(y_val, y_val_pred)
        val_acc_scores.append(val_acc)

        # 计算平均类别精度（macro平均的precision）
        val_avg_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
        val_precision_scores.append(val_avg_precision)

    # 找到最佳K值（基于验证集准确率）
    best_idx = np.argmax(val_acc_scores)
    best_k = k_values[best_idx]
    best_val_acc = val_acc_scores[best_idx]
    best_val_avg_precision = val_precision_scores[best_idx]

    # 使用最佳K值训练最终模型
    neigh_best = KNeighborsClassifier(n_neighbors=best_k)
    neigh_best.fit(x_train, y_train)

    # 训练集预测和评估
    y_train_pred = neigh_best.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    return {
        'train_accuracy': train_acc,
        'val_accuracy': best_val_acc,
        'val_avg_precision': best_val_avg_precision,
        'best_k': best_k,
        'val_scores': val_acc_scores,
        'val_precision_scores': val_precision_scores
    }


if __name__ == '__main__':
    path = '../ckgs/ablation/act_layer/ssl_act_layer_0.5253.pt'
    # path = '../ckgs/ssl_bil_layer/bil_layer_best_0.7019.pt'
    # path = '../ckgs/ssl_m1_reg/m1_reg_best_0.6538.pt'
    config = {
        'name': 'new_BIL',
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
        'data': {
            'root': r'D:\PycharmProjects\B1\dataset\act',
            'path': r'D:\Dataset\Neuron\act',
            'keep_node': 512,
            'batch_size': 32,
            'num_workers': 4 if os.name != 'nt' else 0
        },
    }
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    # model = byol_gg(Encoder, Predictor).to(device)
    state_dict = torch.load(path, map_location=device)

    model.online_encoder.load_state_dict(state_dict)
    model.eval()


    eval_transform = T.Compose([
        # NewBranchCutTransform(keep_nodes=32, protected_nodes=[0],
        #                       allow_disconnect=False, enable_branch_cut=False, max_branch=0),
        AddShortestPathPE(anchor_selection='kmeans', num_anchors=20, attr_name='pe'),
        AddHeatKernelPE_v2(time=[0.1, 1.0, 10], diag_attr_name='hk_pe', full_attr_name='bias'),
    ])
    #
    evel_data = ACT(
        root=config['data']['root'],
        path=config['data']['path'],
        type='layer',
        data_name='act',
        keep_node=512,
        transform=eval_transform
    )
    #
    # evel_data = NeuronDataset(
    #     root=config['data']['root'],
    #     path=config['data']['path'],
    #     data_name='jml',
    #     keep_node=100,
    #     transform=eval_transform
    # )
    labels = [data.layer_y.item() for data in evel_data]
    train_indices, val_indices = train_test_split(
        range(len(evel_data)),
        test_size=0.1,
        # stratify=labels,
        random_state=12
    )
    print(train_indices)
    print(val_indices)

    train_dataset = evel_data.index_select(train_indices)
    val_dataset = evel_data.index_select(val_indices)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])
    # Subset


    print(len(train_dataset))
    print(len(val_dataset))
    # 提取特征
    x_train, y_train = extract_features(model.online_encoder, train_loader, device)
    x_val, y_val = extract_features(model.online_encoder, val_loader, device)

    # 转换为numpy
    x_train_np, y_train_np = x_train.numpy(), y_train.numpy()
    x_val_np, y_val_np = x_val.numpy(), y_val.numpy()

    # 评估
    train_acc, val_acc, best_k = eval_knn_auto_k(x_train_np, y_train_np, x_val_np, y_val_np)
    # results = eval_knn_auto_avg_k(x_train, y_train, x_val, y_val)


    # print(f"最佳K值: {results['best_k']}")
    # print(f"训练集准确率: {results['train_accuracy']:.4f}")
    # print(f"验证集准确率: {results['val_accuracy']:.4f}")
    # print(f"验证集平均类别精度: {results['val_avg_precision']:.4f}")
    # print(f"验证集所有K值的准确率: {results['val_scores']}")
    # print(f"验证集所有K值的平均类别精度: {results['val_precision_scores']}")
    #
    print(f"[KNN] Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Best K={best_k}")
