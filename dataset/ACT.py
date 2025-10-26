#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/24 9:44
# @Author  : ShengPengpeng
# @File    : ACT.py
# @Description :

import warnings
import re
import os
import torch
from tqdm import tqdm
import json
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np
from dataloader.io.read_swc import read_swc
from typing import Callable, List, Optional
from dataloader.NodeFeature import NeuronFeatureCalculator
dendrite_subclasses = {
    "Spiny_1": 0,   "Spiny_2": 1,   "Spiny_3": 2,   "Spiny_4": 3,     "Spiny_5": 4,    "Spiny_6": 5,    "Spiny_7": 6,
    "Spiny_8": 7,   "Spiny_9": 8,   "Spiny_10": 9,  "Spiny_11": 10,   "Spiny_12": 11,  "Spiny_13": 12,  "Spiny_14": 13,
    "Spiny_15": 14, "Spiny_16": 15, "Spiny_17": 16, "Spiny_18": 17,   "Spiny_19": 18,
    "Aspiny_1": 19, "Aspiny_2": 20, "Aspiny_3": 21, "Aspiny_4": 22,   "Aspiny_5": 23,  "Aspiny_6": 24,  "Aspiny_7": 25,
    "Aspiny_8": 26, "Aspiny_9": 27, "Aspiny_10": 28, "Aspiny_11": 29, "Aspiny_12": 30, "Aspiny_13": 31, "Aspiny_14": 32,
    "Aspiny_15": 33, "Aspiny_16": 34, "Aspiny_17": 35, "Aspiny_18": 36, "Aspiny_19": 37,
}
exc_subclass = {
    "Exc_1": 0, "Exc_2": 1, "Exc_3": 2, "Exc_4": 3,
    "Inh_1": 4, "Inh_2": 5, "Inh_3": 6, "Inh_4": 7, "Inh_5": 8, "Inh_6": 9, "Inh_7": 10, "Inh_8": 11, "Inh_9": 12,
    "Inh_10": 13, "Inh_11": 14, "Inh_12": 15, "Inh_13": 16,
}


label_act_project = {
    "spiny": 0,  "aspiny": 1,
}

label_act_layer = {
    "Isocortex_layer2/3": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}

label_jml_layer = {
    "Isocortex_layer2/3": 0,
    'VPM': 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}
label_jml_project = {
    "Isocortex_layer2/3": 0,
    'VPM': 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}

label_bil_project = {'TH_core': 0, 'CTX_ET': 1, 'CTX_IT': 2,'CP_GPe': 3, 'Car3': 4, 'CP_SNr': 5, 'TH_matrix': 6, 'CP_others': 7}

label_bil_layer = {
    "Isocortex_layer2/3": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
    'CP':4,
    'VPM':5
}
# class ACT(InMemoryDataset):
#     def __init__(self,
#                  root: str,
#                  path: str,
#                  split: str,
#                  type: str,
#                  subset: bool = False,
#                  data_name: str = 'act',
#                  keep_node: int = 512,
#                  transform: Optional[Callable] = None,
#                  pre_transform: Optional[Callable] = None,
#                  pre_filter: Optional[Callable] = None,
#                  **kwargs):
#         assert split in ['val', 'train', 'all']
#         self.split = split
#         self.type = type
#         self.path = path
#         self.root = root
#         self.subset = subset
#         self.data_name = data_name
#         self.keep_node = keep_node
#         super().__init__(root, transform, pre_transform, pre_filter)
#
#
#         path = os.path.join(self.processed_dir, f'{self.data_name}_{self.split}_{self.keep_node}.pt')
#         self.data, self.slices = torch.load(path)
#
#
#     @property
#     def processed_file_names(self) -> List[str]:
#         """处理后的文件名"""
#         return [f'{self.data_name}_train_{self.keep_node}.pt',
#                 f'{self.data_name}_val_{self.keep_node}.pt',
#                 f'{self.data_name}_all_{self.keep_node}.pt']
#
#     def process(self):
#         """处理原始数据并转换为PyG格式"""
#         # 确保processed目录存在
#         os.makedirs(self.processed_dir, exist_ok=True)
#         print(self.processed_dir)
#         split_json_path = os.path.join(self.root, f'{self.data_name}_random_split.json')
#         with open(split_json_path) as f:
#             data_split  = json.load(f)
#
#         data_all = []
#         data_both_labels = []
#         data_type_only = []
#         data_layer_only = []
#         data_unlabeled = []
#
#         metadat = pd.read_csv(os.path.join(self.root, f'{self.data_name}_info_swc_10folds.csv'),  encoding='utf-8')
#         for split_type in ['train', 'val', 'all']:
#             data_all.clear()
#             data_both_labels.clear()
#             data_type_only.clear()
#             data_layer_only.clear()
#             data_unlabeled.clear()
#             data_list = []
#             for swc_path in tqdm(data_split[split_type], desc=f'Processing {split_type} neurons'):
#                 swc = os.path.join(self.path, self.data_name, 'swc', swc_path)
#
#             # 构造子目录路径（假设每个文件项是一个子目录）
#             # 读取和预处理神经元形态
#                 morphology = read_swc(swc)
#
#                 morphology.strip_type(2)
#                 sparse_morph = morphology.sparsify(threshold=0.1, target_num=self.keep_node)
#
#             # 计算特征
#                 calculator = NeuronFeatureCalculator(swc_data=sparse_morph.compartment_index, mode='root_scale')
#                 node_features = calculator.node_features
#                 edge_features = calculator.edge_features
#
#             # 构建无向边索引 (用于图神经网络)
#                 directed_edges = []
#                 edge_weights = []
#                 edge_attrs = []
#
#                 for (src, tgt), feat in edge_features.items():
#                     directed_edges.append([src, tgt])  # 只保留原始方向
#                     length = feat['length']
#                     level = feat.get('level', 0)
#                     edge_weights.append(length)
#                     edge_attrs.append(level)
#
#                 edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
#                 edge_weight = torch.tensor(edge_weights, dtype=torch.float)
#                 edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
#
#                 # 构建节点位置和特征
#                 node_ids = sorted(node_features.keys())
#                 pos_list = [node_features[nid]['pos'] for nid in node_ids]
#
#                 feature_keys = [k for k in node_features[node_ids[0]].keys()
#                                 if k != 'pos' and not isinstance(node_features[node_ids[0]][k], list)]
#
#                 x_list = []
#                 for nid in node_ids:
#                     feat_values = []
#                     for k in feature_keys:
#                         val = node_features[nid][k]
#                         feat_values.append(float(val))
#                     x_list.append(feat_values)
#
#
#                 # 构建分支索引
#                 num_nodes = len(node_ids)
#                 G = nx.Graph()
#                 G.add_nodes_from(range(num_nodes))  # 确保所有节点都加入图中
#                 edges = edge_index.t().tolist()
#                 G.add_edges_from(edges)
#
#                 if nx.number_connected_components(G) > 1:
#                     print(f"⚠️ 图不连通！{swc}")
#
#                 branch_index = build_branch_index_from_edges(num_nodes, edge_index)
#                 if branch_index is None or branch_index.numel() == 0:  # 检查None和空张量两种情况
#                     print(f"警告: 分支索引构建失败，跳过当前处理 {swc}")
#                     continue
#
#
#                 layer_dict = globals().get(f'label_{self.data_name}_layer')
#                 type_dict = globals().get(f'label_{self.data_name}_project')
#                 layer = torch.tensor([-1], dtype=torch.long)  # 默认无效标签
#                 type = torch.tensor([-1], dtype=torch.long)  # 默认无效标签
#                 swc_name = os.path.basename(swc_path)
#
#                 if self.data_name == 'bil':
#                     first, last = swc_name.find('_'), swc_name.rfind('_')
#                     search_value = swc_name[first + 1:last]
#                     query, feature = 'specimen__id', 'Subclass_or_type'
#                     # 获取标签（确保标签字典存在）
#                 elif self.data_name == 'act':
#                     search_value = swc_name
#                     query, feature = 'swc__fname', 'tag__dendrite_type'
#                 elif self.data_name == 'jml':
#                     search_value = swc_name
#                     query, feature = 'swc__fname', 'structure__acronym'
#                 else:
#                     print(f"[Error] 未支持的数据集: {self.data_name}")
#
#                 match = metadat.loc[metadat[query] == search_value]
#                 if match.empty:
#                     print(f"[Warning] {query}={search_value} 未在 metadata 中找到")
#
#                 # Layer 标签
#                 layer_val = match['structure_merge__acronym'].values[0]
#                 if isinstance(layer_val, str) and layer_val in layer_dict:
#                     layer = torch.tensor([layer_dict[layer_val]], dtype=torch.long)
#                 else:
#                     print(f"[Warning] 层标签 '{layer_val}' 不在 layer_dict 中")
#
#                 # Type 标签（若有）
#                 if feature:
#                     type_val = match[feature].values[0]
#                     if isinstance(type_val, str) and type_val in type_dict:
#                         type = torch.tensor([type_dict[type_val]], dtype=torch.long)
#                     else:
#                         print(f"[Warning] 类型标签 '{type_val}' 不在 type_dict 中")
#
#
#             # 创建Data对象
#                 data = Data(
#                     id=os.path.basename(swc_path),
#                     x=torch.tensor(x_list, dtype=torch.float),
#                     pos=torch.tensor(pos_list, dtype=torch.float),
#                     edge_index=edge_index,
#                     edge_weight=edge_weight,
#                     edge_attr=edge_attr,
#                     type_y= type ,  # 确保标签是标量张量
#                     layer_y= layer,
#                     branch_index=branch_index,
#                 # directed_edge_index=directed_edge_index  # 保存有向边供后续使用
#                 )
#
#                 if self.pre_filter is not None and not self.pre_filter(data):
#                     continue
#
#                 if self.pre_transform is not None:
#                     data = self.pre_transform(data)
#             # if data.x.shape[0] != self.keep_node:
#             #     print(data)
#                 if data.x.shape[0] == self.keep_node:
#                     if data.type_y.item() != -1:
#                         data_list.append(data)
#                     else:
#                         print(f"[Skipped] {data.id} 的类型标签为 -1，跳过保存")
#
#                 type_valid = data.type_y.item() != -1
#                 layer_valid = data.layer_y.item() != -1
#
#                 if type_valid and layer_valid:
#                     data_both_labels.append(data)
#                 elif type_valid:
#                     data_type_only.append(data)
#                 elif layer_valid:
#                     data_layer_only.append(data)
#                 else:
#                     data_unlabeled.append(data)
#
#             # 保存各类数据
#             def save_data_list(name, dlist):
#                 if len(dlist) > 0:
#                     path = os.path.join(self.processed_dir, f'{self.data_name}_{split_type}_{self.keep_node}_{name}.pt')
#                     torch.save(self.collate(dlist), path)
#                     print(f"✅ 已保存 {len(dlist)} 个 {name} 样本 -> {os.path.basename(path)}")
#
#             save_data_list('all', data_all)
#             save_data_list('type', data_type_only)
#             save_data_list('layer', data_layer_only)



class ACT(InMemoryDataset):
    def __init__(self,
                 root: str,
                 path: str,
                 type: str = 'all',  # 支持 'type' / 'layer' / 'all'
                 subset: bool = False,
                 data_name: str = 'act',
                 keep_node: int = 512,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs):
        assert type in ['type', 'layer', 'all']
        self.type = type
        self.path = path
        self.root = root
        self.subset = subset
        self.data_name = data_name
        self.keep_node = keep_node
        super().__init__(root, transform, pre_transform, pre_filter)

        # 文件名统一：不再按 split 存储
        path = os.path.join(self.processed_dir, f'{self.data_name}_{self.keep_node}_{self.type}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.data_name}_{self.keep_node}_{t}.pt'
                for t in ['all', 'type', 'layer']]

    def pre_filter(data):
        return data.x.size(0) == 256

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        metadat = pd.read_csv(os.path.join(self.root, f'{self.data_name}_info_swc_10folds.csv'), encoding='utf-8')

        layer_dict = globals().get(f'label_{self.data_name}_layer', {})
        type_dict = globals().get(f'label_{self.data_name}_project', {})

        # 扫描整个 SWC 文件夹
        swc_dir = os.path.join(self.path, self.data_name, 'swc')
        swc_paths = sorted([f for f in os.listdir(swc_dir) if f.endswith('.swc')])

        data_all, data_type_only, data_layer_only = [], [], []
        for swc_file in tqdm(swc_paths, desc=f'Processing {self.data_name} neurons'):
            swc = os.path.join(swc_dir, swc_file)
            morphology = read_swc(swc)
            morphology.strip_type(2)
            sparse_morph = morphology.sparsify(threshold=0.1, target_num=self.keep_node)

            calculator = NeuronFeatureCalculator(swc_data=sparse_morph.compartment_index)
            node_features = calculator.node_features
            edge_features = calculator.edge_features


            directed_edges, edge_weights, edge_attrs = [], [], []
            for (src, tgt), feat in edge_features.items():
                # 边1: src -> tgt
                directed_edges.append([src, tgt])
                edge_weights.append(np.mean(feat['length']))  # 长度取平均
                edge_attrs.append(np.mean(feat['level']))  # level取平均

                # 边2: tgt -> src（对称）
                directed_edges.append([tgt, src])
                edge_weights.append(np.mean(feat['length']))
                edge_attrs.append(np.mean(feat['level']))

            edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)

            node_ids = sorted(node_features.keys())
            pos_list = [node_features[nid]['pos'] for nid in node_ids]
            x_list = [[float(node_features[nid][k]) for k in node_features[nid] if k not in ['pos']] for nid in
                      node_ids]

            # # G = nx.Graph(edge_index)
            # num_nodes = int(edge_index.max()) + 1
            # G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
            # if nx.number_connected_components(G) > 1:
            #     print(swc_file)
            #     print(f"❌ 图不连通，包含 {nx.number_connected_components(G)} 个子图")
            #     for i, comp in enumerate(nx.connected_components(G)):
            #         print(f"  Component {i + 1} ({len(comp)} nodes): {sorted(comp)}")
            #
            #     continue

            layer = torch.tensor([-1], dtype=torch.long)
            type = torch.tensor([-1], dtype=torch.long)
            swc_name = os.path.basename(swc_file)

            if self.data_name == 'bil':
                first, last = swc_name.find('_'), swc_name.rfind('_')
                search_value = swc_name[first + 1:last]
                query, feature = 'specimen__id', 'Subclass_or_type'
            elif self.data_name == 'act':
                search_value = swc_name
                query, feature = 'swc__fname', 'tag__dendrite_type'
            elif self.data_name == 'jml':
                search_value = swc_name
                query, feature = 'swc__fname', 'structure_merge__acronym'
            else:
                continue

            match = metadat.loc[metadat[query] == search_value]
            if not match.empty:
                layer_val = match['structure_merge__acronym'].values[0]
                if isinstance(layer_val, (str, int)) and str(layer_val) in layer_dict:
                    layer = torch.tensor([layer_dict[str(layer_val)]], dtype=torch.long)
                if feature:
                    type_val = match[feature].values[0]
                    if isinstance(type_val, (str, int)) and str(type_val) in type_dict:
                        type = torch.tensor([type_dict[str(type_val)]], dtype=torch.long)

            data = Data(
                id=swc_name,
                x=torch.tensor(x_list, dtype=torch.float),
                pos=torch.tensor(pos_list, dtype=torch.float),
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                type_y=type,
                layer_y=layer,
                # branch_index=branch_index
            )

            G = to_networkx(data, to_undirected=True)
            if not nx.is_connected(G):
                print(f"❌ 图不连通，包含 {nx.number_connected_components(G)} 个子图")
                for i, comp in enumerate(nx.connected_components(G)):
                    print(f"  Component {i + 1} ({len(comp)} nodes): {sorted(comp)}")
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # 检查节点数是否仍符合要求（可能在transform中改变）
            # if data.x.size(0) != self.keep_node:
            #     print(f'Skipping {swc_file}')
            #     continue

            # 根据标签有效性添加到对应数据集
            type_valid = data.type_y.item() != -1
            layer_valid = data.layer_y.item() != -1

            if type_valid:
                data_type_only.append(data)
            if layer_valid:
                data_layer_only.append(data)
            # if type_valid or layer_valid:
            data_all.append(data)

        # 保存数据集
        def save(name, dlist):
            path = os.path.join(self.processed_dir, f'{self.data_name}_{self.keep_node}_{name}.pt')
            torch.save(self.collate(dlist), path)
            print(f"✅ Saved {len(dlist)} {name} samples")

        save('all', data_all)
        save('type', data_type_only)
        save('layer', data_layer_only)


if __name__ == '__main__':

    import torch_geometric.transforms as T
    from dataloader.SPE import AddShortestPathPE
    from dataloader.augment import NewBranchCutTransform
    from augment import NewBranchCutTransform

    path = 'D:/Dataset/Neuron/'
    root = '../dataset/jml'


    Transform = T.Compose([
        # RandomJitterNeurite(noise_std=0.01, translate=0.05, rotate=True, flip=True, seed=42),
        # NewBranchCutTransform(keep_nodes=512, protected_nodes=[0], allow_disconnect=False,
        #                       enable_branch_cut=True, max_branch=35),
        AddShortestPathPE(anchor_selection='kmeans', num_anchors=20, attr_name='pe'),
    ])

    dataset_1  = ACT(
    root=root,
    path=path,
    type='layer',
    data_name='jml',
    keep_node=100,
    transform=Transform,
)

    # dataset_2 = ACT(root=root, path=path ,  data_name='act', keep_node=512,pre_transform=Transform)
    # dataset_3 = ACT(root=root, path=path ,  data_name='act', keep_node=256,pre_transform=Transform)
    #
    #
    # labeled_dataset = [data for data in dataset if data.type_y.item() != -1]
    # unlabeled_dataset = [data for data in dataset if data.type_y.item() == -1]
    #
    # num_graphs = len(labeled_dataset)
    # perm = torch.randperm(num_graphs)
    # train_idx = perm[:int(0.8 * num_graphs)]
    # val_idx = perm[int(0.8 * num_graphs):]
    # train_dataset = [labeled_dataset[i] for i in train_idx]
    # val_dataset = [labeled_dataset[i] for i in val_idx]
    #
    # unsup_loader = DataLoader(unlabeled_dataset + train_dataset, batch_size=16, shuffle=True)  # 训练阶段忽略标签
    # sup_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)  # probe 阶段
    #
    # print(f"Num labeled: {len(labeled_dataset)} | Num unlabeled: {len(unlabeled_dataset)}")
    # print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(len(dataset_1))
    for i, data in enumerate(dataset_1):
        print(data.layer_y, data.type_y)
        # if data.x.size(0) != 256:
        #     print(f"❌ 图 {i} 的节点数为 {data.x.size(0)}，不等于 256")
        #     print(data)

    # for data in dataset_1:
    #     print( data.x, data.pos , data.edge_index, data.edge_attr, data.type_y, data.reg_y, data.branch_index)
    # print(act_dataset[0].edge_index)
    # print(act_dataset[0].id)
