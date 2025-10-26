

from tqdm import tqdm
from torch_geometric.data import database
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from collections import OrderedDict
import umap  # pip install umap-learn
import pandas as pd
import torch

from dataloader.loader import NeuronDataset
from torch_geometric.loader import DataLoader
from model.ssl_3 import Transformer_encoder, MLP_Predictor, byol_gg
import os
import torch_geometric.transforms as T
from dataloader.augment import NewBranchCutTransform
from dataloader.HKPE import AddHeatKernelPE_v2
from dataloader.SPE import AddShortestPathPE
from sklearn.model_selection import train_test_split
from dataloader.ACT import ACT
import numpy as np
from umap import UMAP


# LABEL_MAP = {"Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6"}

# LABEL_MAP = {
#     "Exc_1", "Exc_2", "Exc_3", "Exc_4",
#     "Inh_1", "Inh_2", "Inh_3", "Inh_4", "Inh_5", "Inh_6", "Inh_7", "Inh_8", "Inh_9",
#     "Inh_10", "Inh_11", "Inh_12", "Inh_13",
# }

# LABEL_MAP = {"Exc_1", "Exc_2","Inh_1", "Inh_2", "Inh_3", "Inh_4"}
LABEL_MAP = {'Sst', 'Vip', 'Pvalb', 'Lamp5', 'L4', 'NP', 'L6b', 'L6_IT', 'L5_IT', 'L6_CT', 'L5_CF' , 'L23_IT'}

# LABEL_MAP = {
#         "Spiny_1",   "Spiny_2",   "Spiny_3",   "Spiny_4",     "Spiny_5",    "Spiny_6",    "Spiny_7",
#         "Spiny_15", "Spiny_16", "Spiny_17", "Spiny_18",   "Spiny_19",
#         "Aspiny_1", "Aspiny_2", "Aspiny_3", "Aspiny_4",   "Aspiny_5",  "Aspiny_6",  "Aspiny_7",
#         "Aspiny_8", "Aspiny_9", "Aspiny_10", "Aspiny_11", "Aspiny_12", "Aspiny_13", "Aspiny_14",
#         "Aspiny_15", "Aspiny_16", "Aspiny_17", "Aspiny_18", "Aspiny_19"
#     }

# LABEL_MAP = {"spiny", "aspiny",}
# LABEL_MAP =  {'Exc', 'Inh'}
# LABEL_MAP = {
#         "Spiny_1",   "Spiny_2",   "Spiny_3",   "Spiny_4",     "Spiny_5",    "Spiny_6",    "Spiny_7",
#         "Aspiny_1", "Aspiny_2", "Aspiny_3", "Aspiny_4",   "Aspiny_5",  "Aspiny_6"
#     }

def extract_embedding(model,
                      loader,
                      csv_file,
                      device,
                      id_col='swc__fname',
                      # label_col='e-type',
                      label_col='me1-type',
                      # label_col='structure_merge__acronym',
                      # label_col = 'tag__dendrite_type',
                      save_path="./database.npy"):
    """
    提取特征并保存为 {label_name/sample_id: feature} 的字典
    """

    df_labels = pd.read_csv(csv_file, encoding='gbk')
    if id_col not in df_labels.columns:
        raise ValueError(f"ID列 '{id_col}' 不存在于CSV文件中")
    if label_col not in df_labels.columns:
        raise ValueError(f"标签列 '{label_col}' 不存在于CSV文件中")

    # 创建ID到标签的映射字典
    id_to_label = dict(zip(df_labels[id_col], df_labels[label_col]))
    print(f"从CSV文件加载了 {len(id_to_label)} 个样本的标签映射")


    # 设置模型为评估模式（关闭dropout、batch norm等训练专用层）
    model.eval()
    feature_db = {}  # 创建空字典存储特征

    for i, data in enumerate(tqdm(loader, desc="Extracting features")):
        data = data.to(device)

        with torch.no_grad():  # 添加无梯度计算，节省内存
            feats = model(data)

        feats = feats.cpu().numpy()

        if isinstance(data.id, list):
            sample_ids = data.id
        else:
            # 假设data.id是包含多个ID的张量或数组
            sample_ids = data.id.cpu().numpy() if hasattr(data.id, 'cpu') else data.id

        # 确保sample_ids是可迭代的
        if not isinstance(sample_ids, (list, np.ndarray)):
            sample_ids = [sample_ids]

        # 遍历当前批次中的每个样本
        for j in range(feats.shape[0]):
            if j < len(sample_ids):
                sample_id = str(sample_ids[j])  # 确保ID是字符串
            else:
                sample_id = f"batch_{i}_sample_{j}"

            # 从CSV文件中获取标签
            if sample_id in id_to_label:
                label_name = id_to_label[sample_id]
            else:
                print(f"警告: 样本ID '{sample_id}' 未在CSV文件中找到，跳过该样本")
                continue  # 或者使用默认标签

            # 创建格式为 "标签名/样本ID" 的唯一key
            key = f"{label_name}/{sample_id}"
            feature_db[key] = feats[j]

    # 将特征字典保存为npy文件
    np.save(save_path, feature_db)
    print(f"Feature database saved to {save_path}, total {len(feature_db)} samples")
    return feature_db



def extract_features(model, loader, device, save_path="./database.npy"):
    """
    提取特征并保存为 {label_name/sample_id: feature} 的字典
    """
    model.eval()
    feature_db = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Extracting features")):
            # 将数据移动到设备
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            else:
                # 如果batch是元组或列表
                batch = [item.to(device) if hasattr(item, 'to') else item for item in batch]

            # 获取特征
            feats = model(batch)

            # 安全转换为numpy
            feats = feats.cpu().detach().numpy()

            # 获取标签（根据你的数据格式调整）
            labels = batch.layer_y.cpu().numpy()

            # 处理每个样本
            for j in range(feats.shape[0]):
                label_id = int(labels[j])
                label_name = LABEL_MAP.get(label_id, f"unknown_{label_id}")

                # 生成唯一ID
                if hasattr(batch, "id"):
                    if isinstance(batch.id, (list, tuple)) and j < len(batch.id):
                        sample_id = batch.id[j]
                    else:
                        sample_id = batch.id

                key = f"{label_name}/{sample_id}"
                feature_db[key] = feats[j]

    # 保存特征数据库
    np.save(save_path, feature_db)
    print(f"Feature database saved to {save_path}, total {len(feature_db)} samples")
    return feature_db



def ReturnLabel(database):

    # label7 = {"spiny": 0, "aspiny": 1 , }

    # label7 = {
    #     "Exc_1": 0, "Exc_2": 1, "Exc_3": 2, "Exc_4": 3,
    #     "Inh_1": 4, "Inh_2": 5, "Inh_3": 6, "Inh_4": 7, "Inh_5": 8, "Inh_6": 9, "Inh_7": 10, "Inh_8": 11, "Inh_9": 12,
    #     "Inh_10": 13, "Inh_11": 14, "Inh_12": 15, "Inh_13": 16,
    # }

    # label7 = {"Exc_1": 0, "Exc_2": 1,"Inh_1": 2, "Inh_2": 3, "Inh_3": 4, "Inh_4": 5}
    label7 = {'Sst':0, 'Vip':1, 'Pvalb':2, 'Lamp5':3,
              'L4':4, 'NP':5, 'L6b':6, 'L6_IT':7, 'L5_IT':8, 'L6_CT':9, 'L5_CF':10 , 'L23_IT':11}
    #
    # label7 = {
    #     "Spiny_1": 0,   "Spiny_2": 1,   "Spiny_3": 2,   "Spiny_4": 3,     "Spiny_5": 4,    "Spiny_6": 5,    "Spiny_7": 6,
    #     "Aspiny_1": 7, "Aspiny_2": 8, "Aspiny_3": 9, "Aspiny_4": 10,   "Aspiny_5": 11,  "Aspiny_6": 12,  "Aspiny_7": 13,
    # }
    # label7 = {
    #     "Spiny_1": 0,   "Spiny_2": 1,   "Spiny_3": 2,   "Spiny_4": 3,     "Spiny_5": 4,    "Spiny_6": 5,    "Spiny_7": 6,
    #     "Spiny_8": 7,   "Spiny_9": 8,   "Spiny_10": 9,  "Spiny_11": 10,   "Spiny_12": 11,  "Spiny_13": 12,  "Spiny_14": 13,
    #     "Spiny_15": 14, "Spiny_16": 15, "Spiny_17": 16, "Spiny_18": 17,   "Spiny_19": 18,
    #     "Aspiny_1": 19, "Aspiny_2": 20, "Aspiny_3": 21, "Aspiny_4": 22,   "Aspiny_5": 23,  "Aspiny_6": 24,  "Aspiny_7": 25,
    #     "Aspiny_8": 26, "Aspiny_9": 27, "Aspiny_10": 28, "Aspiny_11": 29, "Aspiny_12": 30, "Aspiny_13": 31, "Aspiny_14": 32,
    #     "Aspiny_15": 33, "Aspiny_16": 34, "Aspiny_17": 35, "Aspiny_18": 36, "Aspiny_19": 37,
    # }
    # label7 = {"Isocortex_layer23": 0,"Isocortex_layer4": 1,"Isocortex_layer5": 2,"Isocortex_layer6": 3}

    label_list = []
    valid_keys = []  # 记录有效的key

    for swc in database.keys():
        label_name = swc[0:swc.find('/')]
        print(label_name)
        if label_name in label7:
            label_list.append(label7[label_name])
            valid_keys.append(swc)
        else:
            print(f"跳过未知标签: {label_name}")

    return np.array(label_list), valid_keys




def Tsne(database, n_components=2, perplexity=30, figsize=None):


    # label7 = ["Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6"]
    # label7 = [
    #     "Spiny_1",   "Spiny_2",   "Spiny_3",   "Spiny_4",     "Spiny_5",    "Spiny_6",    "Spiny_7",
    #     "Spiny_8",   "Spiny_9",   "Spiny_10",  "Spiny_11",   "Spiny_12",  "Spiny_13",  "Spiny_14",
    #     "Spiny_15", "Spiny_16", "Spiny_17", "Spiny_18",   "Spiny_19",
    #     "Aspiny_1", "Aspiny_2", "Aspiny_3", "Aspiny_4",   "Aspiny_5",  "Aspiny_6",  "Aspiny_7",
    #     "Aspiny_8", "Aspiny_9", "Aspiny_10", "Aspiny_11", "Aspiny_12", "Aspiny_13", "Aspiny_14",
    #     "Aspiny_15", "Aspiny_16", "Aspiny_17", "Aspiny_18", "Aspiny_19"
    # ]

    # label7 = [
    #     "Spiny_1",   "Spiny_2",   "Spiny_3",   "Spiny_4",     "Spiny_5",    "Spiny_6",    "Spiny_7",
    #     "Aspiny_1", "Aspiny_2", "Aspiny_3", "Aspiny_4",   "Aspiny_5",  "Aspiny_6",
    # ]
    # label7 = [ "Exc_1", "Exc_2", "Inh_1", "Inh_2", "Inh_3", "Inh_4"]

    label7 = ['Sst', 'Vip', 'Pvalb', 'Lamp5', 'L4', 'NP', 'L6b', 'L6_IT', 'L5_IT', 'L6_CT', 'L5_CF' , 'L23_IT']

    # label7 = ['spiny', 'aspiny']
    # label7 = [    "Exc_1", "Exc_2", "Exc_3", "Exc_4",
    # "Inh_1", "Inh_2", "Inh_3", "Inh_4", "Inh_5", "Inh_6", "Inh_7", "Inh_8", "Inh_9",
    # "Inh_10", "Inh_11", "Inh_12", "Inh_13",]
    # label7 = ['Inh', 'Exc' ]


    # 扩展的颜色列表（37种颜色）
    extended_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
        '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39',
        '#7b4173', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252',
        '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94',
        '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194'
    ]

    # 标记列表
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h',
               'X', 'P', 'd', '|', '_', '+', 'x', '1', '2', '3', '4']

    database = np.load(database, allow_pickle=True).item()
    label, valid_keys = ReturnLabel(database)
    feature = [database[key] for key in valid_keys]
    feature = np.array(feature)

    print(f"有效样本数量: {feature.shape[0]}")
    print(f"特征维度: {feature.shape[1]}")
    unique_labels, counts = np.unique(label, return_counts=True)
    print(f"标签分布: {dict(zip([label7[i] for i in unique_labels], counts))}")

    # t-SNE降维
    tsne = manifold.TSNE(n_components=n_components,
                         init='pca',
                         random_state=13,
                         perplexity=perplexity)
    X_tsne = tsne.fit_transform(feature)

    # 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # 设置图形大小
    if figsize is None:
        figsize = (10, 8) if n_components == 2 else (12, 10)

    if n_components == 2:
        # 2D可视化
        plt.figure(figsize=figsize)
        for i, label_value in enumerate(np.unique(label)):
            # 计算颜色和标记索引
            color_idx = i % len(extended_colors)
            marker_idx = (i // len(extended_colors)) % len(markers)

            mask = label == label_value
            plt.scatter(X_norm[mask, 0], X_norm[mask, 1],
                        c=[extended_colors[color_idx]],
                        marker=markers[marker_idx],
                        label=label7[label_value],
                        alpha=0.8, s=60,
                        edgecolors='black', linewidth=0.3)

            # plt.scatter(X_norm[mask, 0], X_norm[mask, 1],
            #             color=plt.cm.tab20b(label_value),
            #             label=label7[label_value],
            #             alpha=0.8, s=50, edgecolors='w', linewidth=0.5)

        plt.legend(loc='upper right', fontsize=10, frameon=True)
        # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center',
        #            ncol=5, fontsize=10, frameon=True)
        # plt.title('Embeddings Cortical Regions Visualization', size=16)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

    else:
        # 3D可视化
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for i, label_value in enumerate(np.unique(label)):

            color_idx = i % len(extended_colors)
            marker_idx = (i // len(extended_colors)) % len(markers)
            mask = label == label_value
            ax.scatter(X_norm[mask, 0], X_norm[mask, 1], X_norm[mask, 2],
                       color =[extended_colors[color_idx]],
                       marker=markers[marker_idx],
                       label=label7[label_value],
                       alpha=0.8, s=40, edgecolors='w', linewidth=0.3)

        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
        # ax.set_title('3D t-SNE Visualization', size=16)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

    plt.tight_layout()
    plt.show()

    return X_tsne



def Umap_visualization(database):
    """单独的UMAP可视化"""
    # label7 =  ['TH_core', 'CTX_ET', 'CTX_IT', 'CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others']
    label7 = ['Sst', 'Vip', 'Pvalb', 'Lamp5', 'L4', 'NP', 'L6b', 'L6_IT', 'L5_IT', 'L6_CT', 'L5_CF' , 'L23_IT']
    # label7 = ["Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6"]
    # label7 = ["spiny", "aspiny"]

    database = np.load(database, allow_pickle=True).item()
    label, valid_keys = ReturnLabel(database)
    feature = [database[key] for key in valid_keys]
    feature = np.array(feature)

    # 确保特征和标签长度一致
    if len(feature) != len(label):
        print(f"特征数量 ({len(feature)}) 和标签数量 ({len(label)}) 不匹配，进行调整")
        min_len = min(len(feature), len(label))
        feature = feature[:min_len]
        label = label[:min_len]

    print(f"特征形状: {feature.shape}")
    print(f"标签形状: {label.shape}")
    print(f"标签值范围: {np.unique(label)}")

    # 优化UMAP参数
    umap_model = UMAP(
        n_components=2,
        random_state=42,  # 使用42避免警告
        n_neighbors=15,
        min_dist=0.1,
        n_epochs=200,
        verbose=True
    )

    X_umap = umap_model.fit_transform(feature)

    x_min, x_max = X_umap.min(0), X_umap.max(0)
    X_norm = (X_umap - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 8))

    # 方法一：使用循环（修复版）
    for i in range(X_norm.shape[0]):
        # 确保label[i]是整数
        label_idx = int(label[i])  # 显式转换为整数
        plt.scatter(X_norm[i, 0], X_norm[i, 1],
                    color=plt.cm.Set2(label_idx),
                    label=label7[label_idx] if i == 0 else "",  # 只在第一个点添加标签
                    alpha=0.7, s=50)

    # 方法二：更高效的方式（推荐）
    # for label_value in np.unique(label):
    #     mask = label == label_value
    #     plt.scatter(X_norm[mask, 0], X_norm[mask, 1],
    #                color=plt.cm.Set2(label_value),
    #                label=label7[label_value],
    #                alpha=0.7, s=50)

    # 创建图例（避免重复标签）
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='upper right',
               shadow=False,
               frameon=False,
               handletextpad=0.2,
               fontsize=10,
               bbox_to_anchor=(1.15, 1.0))  # 防止图例遮挡

    plt.title('UMAP Visualization', size=16)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # path = '../ckgs/ssl_act_layer/act_layer_best_0.6263.pt'
    path = '../ckgs/ssl_act_layer/act_layer_best_0.5960.pt'

    # path = '../ckgs/ssl_act_layer/act_layer_train_0.7215.pt'
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

    model = byol_gg(Encoder, Predictor).to(device)
    state_dict = torch.load(path, map_location=device)
    model.online_encoder.load_state_dict(state_dict)
    model.eval()


    eval_transform = T.Compose([
        AddShortestPathPE(anchor_selection='kmeans', num_anchors=20, attr_name='pe'),
        AddHeatKernelPE_v2(time=[0.1, 1.0, 10], diag_attr_name='hk_pe', full_attr_name='bias'),
    ])


    # eval_data = NeuronDataset(
    #     root=config['data']['root'],
    #     path=config['data']['path'],
    #     data_name='m1',
    #     keep_node=32,
    #     transform=eval_transform
    # )

    eval_data = ACT(
        root=config['data']['root'],
        path=config['data']['path'],
        type='all',
        data_name='act',
        keep_node=256,
        transform=eval_transform
    )


    eval_loader = DataLoader(eval_data, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    # 提取特征

    # 提取特征并保存
    # feature_db = extract_features(model.online_encoder, eval_loader, device, save_path="act_type_feature_database.npy")
    # feature_db = extract_embedding(
    #     model=model.online_encoder,
    #     loader=eval_loader,
    #     device=device,
    #     csv_file='../dataset/act/B_with_labels.csv',
    #     save_path="act_e_feature_database.npy"
    # )

    # path = '../image/act/act_me_feature_database.npy'     # 可视化
    Umap_visualization('act_e_feature_database.npy')
    # Tsne('act_e_feature_database.npy', n_components=2)

    # 3D可视化
    # Tsne('act_exc_feature_database.npy', n_components=3, perplexity=20, figsize=(14, 10))