

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
from model.ssl_4 import byol_denoise
import os
import torch_geometric.transforms as T
from dataloader.augment import NewBranchCutTransform
from dataloader.HKPE import AddHeatKernelPE_v2
from dataloader.SPE import AddShortestPathPE
from sklearn.model_selection import train_test_split
from dataloader.ACT import ACT
import numpy as np
from umap import UMAP
# LABEL_MAP = { 0: "amacrine", 1: "aspiny", 2: "basket", 3: "bipolar", 4: "pyramidal", 5: "spiny", 6: "stellate"}


# LABEL_MAP = {0: "other",1: "tufted",2: "untufted"}
# LABEL_MAP = { 0: "spiny", 1: "aspiny"}
# LABEL_MAP =  {0: 'TH_core', 1: 'CTX_ET', 2:'CTX_IT',3:'CP_GPe', 4:'Car3', 5:'CP_SNr', 6:'TH_matrix', 7:'CP_others'}
# LABEL_MAP = {0:"Isocortex_layer23", 1:"Isocortex_layer4", 2:"Isocortex_layer5", 3:"Isocortex_layer6", 4:'CP', 5:'VPM'}
# LABEL_MAP = {0:"Isocortex_layer23",1:"Isocortex_layer4",2:"Isocortex_layer5",3:"Isocortex_layer6"}



# LABEL_MAP =  {'Prkcd_Grin2c', 'CTX_ET_5','Drd2','CTX_IT_23' ,'Car3','Drd1','CTX_IT_4','CTX_IT_5'}
# LABEL_MAP =  {'TH_core',  'CTX_ET', 'CTX_IT','CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others'}
LABEL_MAP = {'Corticocortical (intratelencephalic)', 'Corticocortical(claustrum-like)', 'Corticofugal(extratelencephalic)', 'Thalamocortical', 'Striatofugal'}

def extract_embedding(model,
                      loader,
                      csv_file,

                      device,
                      id_col='swc__fname',
                      label_col='Subclass_pro',
                      # label_col='Subclass_or_type',
                      # label_col='Subclass_layer',
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
    # label7 = {'amacrine': 0, 'aspiny': 1, 'basket': 2, 'bipolar': 3, 'pyramidal': 4, 'spiny': 5, 'stellate': 6}
    # label7 = {"other": 0, "tufted": 1, "untufted": 2}
    # label7 = {"spiny": 0, "aspiny": 1, }
    # label7 = {"Isocortex_layer23": 0, "Isocortex_layer4": 1, "Isocortex_layer5": 2, "Isocortex_layer6": 3, 'CP':4, 'VPM':5}
    # label7 = {"Isocortex_layer23": 0,"Isocortex_layer4": 1,"Isocortex_layer5": 2,"Isocortex_layer6": 3}
    # label7 = {'TH_core': 0, 'CTX_ET': 1, 'CTX_IT': 2, 'CP_GPe': 3, 'Car3': 4, 'CP_SNr': 5, 'TH_matrix': 6, 'CP_others': 7}
    label7 = {'Corticocortical (intratelencephalic)': 0, 'Corticocortical(claustrum-like)': 1, 'Corticofugal(extratelencephalic)': 2, 'Thalamocortical': 3, 'Striatofugal': 4, }
    # label7 =   {'Prkcd_Grin2c':0, 'CTX_ET_5':1, 'Drd2':2, 'CTX_IT_23':3, 'Car3':4, 'Drd1':5, 'CTX_IT_4':6, 'CTX_IT_5':7}


    # label7 =   {'Prkcd_Grin2c', 'CTX_ET_5', 'Drd2', 'CTX_IT_23', 'Car3', 'Drd1', 'CTX_IT_4', 'CTX_IT_5'}
    # label7 = {'TH_core', 'CTX_ET', 'CTX_IT', 'CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others'}
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


def ReturnLabel_v1(database):
    """修复的ReturnLabel函数，确保返回整数数组"""
    label_list = []
    # label7 = {'TH_core': 0, 'CTX_ET': 1, 'CTX_IT': 2, 'CP_GPe': 3, 'Car3': 4, 'CP_SNr': 5, 'TH_matrix': 6,
    #           'CP_others': 7}
    label7 = {'TH_core': 0, 'CTX_ET': 1, 'CTX_IT': 2, 'CP_GPe': 3, 'Car3': 4, 'CP_SNr': 5, 'TH_matrix': 6, 'CP_others': 7}


    for swc in database.keys():
        label_name = swc[0:swc.find('/')]
        if label_name in label7:
            label_list.append(label7[label_name])
        else:
            # 处理未知标签，跳过或赋予默认值
            print(f"警告: 未知标签 '{label_name}'，已跳过")
            continue

    # 确保返回的是整数numpy数组
    return np.array(label_list, dtype=int)



def baseTsne(database):
    # label7 = ['Amacrine', 'Aspiny', 'Basket', 'Bipolar', 'Pyramidal', 'Spiny', 'Stellate']
    # label7 = ["other", "tufted", "untufted"]
    # label7 = ["spiny", "aspiny", "sparsely spiny"]
    # label7 =  ['TH_core', 'CTX_ET', 'CTX_IT', 'CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others']
    # label7 = ["Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6",'CP','VPM']
    # label7 = ["Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6"]
    label7 =  ['TH_core', 'CTX_ET', 'CTX_IT', 'CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others']
    # label7 = ['Prkcd_Grin2c', 'CTX_ET_5', 'Drd2', 'CTX_IT_23', 'Car3', 'Drd1', 'CTX_IT_4', 'CTX_IT_5']

    database = np.load(database, allow_pickle=True).item()

    # 获取有效的标签和对应的key
    label, valid_keys = ReturnLabel(database)

    # 只使用有效的特征
    feature = [database[key] for key in valid_keys]
    feature = np.array(feature)

    print(f"有效样本数量: {feature.shape[0]}")
    print(f"特征维度: {feature.shape[1]}")
    print(f"标签分布: {np.unique(label, return_counts=True)}")

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=13)
    X_tsne = tsne.fit_transform(feature)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 6))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set2(label[i]),
                    label=label7[label[i]])

    handles, labels_plt = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels_plt, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right',
               shadow=False, frameon=False, handletextpad=0.2, fontsize=10)

    plt.title('t-SNE Visualization (Filtered Labels)', size=16)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def Tsne(database, n_components=2, perplexity=30, figsize=None):
    """
    可配置的t-SNE可视化，支持2D和3D

    参数:
        n_components: 2 或 3，决定可视化维度
        perplexity: t-SNE的perplexity参数
        figsize: 图形大小
    """
    # label7 =  ['TH_core', 'CTX_ET', 'CTX_IT', 'CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others']
    # label7 = ['Prkcd_Grin2c', 'CTX_ET_5', 'Drd2', 'CTX_IT_23', 'Car3', 'Drd1', 'CTX_IT_4', 'CTX_IT_5']
    # label7 = ["Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6",'CP','VPM']
    label7 = ['Corticocortical (intratelencephalic)', 'Corticocortical(claustrum-like)', 'Corticofugal(extratelencephalic)', 'Thalamocortical', 'Striatofugal' ]

    database = np.load(database, allow_pickle=True).item()
    label, valid_keys = ReturnLabel(database)
    feature = [database[key] for key in valid_keys]
    feature = np.array(feature)

    print(f"有效样本数量: {feature.shape[0]}")
    print(f"特征维度: {feature.shape[1]}")
    unique_labels, counts = np.unique(label, return_counts=True)
    print(f"标签分布: {dict(zip([label7[i] for i in unique_labels], counts))}")

    # t-SNE降维
    tsne = manifold.TSNE(n_components=n_components, init='pca',
                         random_state=13, perplexity=perplexity)
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
        for label_value in np.unique(label):
            mask = label == label_value
            plt.scatter(X_norm[mask, 0], X_norm[mask, 1],
                        color=plt.cm.Set2(label_value),
                        label=label7[label_value],
                        alpha=0.8, s=50, edgecolors='w', linewidth=0.5)

        plt.legend(loc='upper right', fontsize=10, frameon=True)
        # plt.title('Embeddings Cortical Regions Visualization', size=16)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

    else:
        # 3D可视化
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for label_value in np.unique(label):
            mask = label == label_value
            ax.scatter(X_norm[mask, 0], X_norm[mask, 1], X_norm[mask, 2],
                       color=plt.cm.Set2(label_value),
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
    label7 =  ['TH_core', 'CTX_ET', 'CTX_IT', 'CP_GPe', 'Car3', 'CP_SNr', 'TH_matrix', 'CP_others']

    # label7 = ["Isocortex_layer23","Isocortex_layer4","Isocortex_layer5","Isocortex_layer6"]
    # label7 = ["spiny", "aspiny"]
    database = np.load(database, allow_pickle=True).item()
    feature = [value for value in database.values()]
    feature = np.array(feature)

    # 获取标签
    label = ReturnLabel_v1(database)

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
    # path = '../ckgs/ssl_act_layer/act_best_0.5859.pt'
    # path = '../ckgs/SSL_bil_type/bil_best_0.8196.pt'
    # path = '../ckgs/ssl_bil_layer/bil_layer_best_0.7019.pt'
    path = '../ckgs/ablation/bil_layer/ssl_bil_train_0.8566.pt'
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
            'root': r'D:\PycharmProjects\B1\dataset\bil',
            'path': r'D:\Dataset\Neuron\bil',
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

    # model = byol_gg(Encoder, Predictor).to(device)
    model = byol_denoise(Encoder, Predictor).to(device)
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
        data_name='bil',
        keep_node=512,
        transform=eval_transform
    )



    # for data in eval_data:
    #     print(data)

    eval_loader = DataLoader(eval_data, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'])

    # 提取特征

    # 提取特征并保存
    # feature_db = extract_features(model.online_encoder, eval_loader, device, save_path="bil_layer_feature_database.npy")
    feature_db = extract_embedding(
        model=model.online_encoder,
        loader=eval_loader,
        device=device,
        csv_file='../dataset/bil/new_file_modified.csv',
        save_path="bil_pro_feature_database.npy"
    )


    # 可视化
    # Umap_visualization('bil_type_feature_database.npy')
    Tsne("bil_pro_feature_database.npy", n_components=2)
    # baseTsne('bil_layer_feature_database.npy')

    # 3D可视化
    # Tsne('bil_tran_feature_database.npy', n_components=3, perplexity=20, figsize=(14, 10))