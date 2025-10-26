#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/21 10:11
# @Author  : ShengPengpeng
# @File    : NodeFeature.py
# @Description :

import warnings

import numpy as np
from typing import Dict, Any, Set, List, Tuple

import networkx as nx
import torch
from collections import deque



class NeuronFeatureCalculator(object):

    def __init__(self,
                 swc_data: Dict[int, Dict[str, Any]],
                 normalize: bool = True,
                 mode: str = 'root_scale'):

        if normalize:
            self.swc_data = self._normalize_coordinates(swc_data, mode)
        else:
            self.swc_data = swc_data

        self.node_features = {}
        self.edge_features = {}

        self._initialize_node_features()
        self._compute_edge_geometry()
        self._compute_topology()
        self._compute_strahler_order()
        self._calculate_edge_levels(mode='child_strahler')


    def _compute_edge_geometry(self):
        # 先计算所有普通边的平均长度（用于自环权重）
        lengths = []
        for nid, node in self.swc_data.items():
            pid = node.get('parent')
            if pid != -1 and pid in self.swc_data:
                length = np.linalg.norm([
                    node['x'] - self.swc_data[pid]['x'],
                    node['y'] - self.swc_data[pid]['y'],
                    node['z'] - self.swc_data[pid]['z']
                ])
                lengths.append(length)
                self.edge_features[(pid, nid)] = {
                    'length': length,
                    'direction': (node['x'] - self.swc_data[pid]['x']) / length if length > 0 else 0.0,
                    'level': 0  # 临时占位，后续由 _calculate_edge_levels 填充
                }

    def _initialize_node_features(self):
        for nid, node in self.swc_data.items():
            self.node_features[nid] = {
                'type': node['type'],
                'radius': node['radius'],
                'pos': [node['x'], node['y'], node['z']],
                'distance_from_root': 0,
                'path_length_to_terminal': 0,
                'is_terminal': 0,
                'is_branch': 0,
                'children_count': 0,
                'subtree_size': 1,
                'strahler_order': 1,
            }
    def _calculate_edge_levels(self, mode: str = 'child_strahler') -> None:
        for (parent_id, child_id), edge_attr in self.edge_features.items():
            parent_node = self.node_features[parent_id]
            child_node = self.node_features[child_id]

            if mode == 'child_strahler':
                level = child_node['strahler_order']
            elif mode == 'parent_strahler':
                level = parent_node['strahler_order']
            elif mode == 'avg_strahler':
                level = 0.5 * (parent_node['strahler_order'] + child_node['strahler_order'])
            elif mode == 'min_strahler':
                level = min(parent_node['strahler_order'], child_node['strahler_order'])
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # 保存到 edge_features 中
            self.edge_features[(parent_id, child_id)]['level'] = level

    def _compute_topology(self):
        root = self._get_root_id()
        dist_root = {root: 0}
        path_to_term = {}
        terminal_nodes = set()

        # BFS: root 到每个节点的距离
        queue = deque([root])
        while queue:
            nid = queue.popleft()
            for cid in self.swc_data[nid].get('children', []):
                elen = self.edge_features.get((nid, cid), {}).get('length', 0)
                dist_root[cid] = dist_root[nid] + elen
                queue.append(cid)

        # 找到终端节点
        for nid, node in self.swc_data.items():
            children = node.get('children', [])
            self.node_features[nid]['children_count'] = len(children)
            self.node_features[nid]['is_terminal'] = int(len(children) == 0)
            self.node_features[nid]['is_branch'] = int(len(children) >= 2)
            if len(children) == 0:
                terminal_nodes.add(nid)
                path_to_term[nid] = 0

        # 反向 DFS 计算到终端的最长路径
        stack = list(terminal_nodes)
        while stack:
            cid = stack.pop()
            pid = self.swc_data[cid].get('parent')
            if pid == -1:
                continue
            elen = self.edge_features.get((pid, cid), {}).get('length', 0)
            new_len = path_to_term[cid] + elen
            if new_len > path_to_term.get(pid, 0):
                path_to_term[pid] = new_len
                stack.append(pid)

        # 子树大小计算
        subtree_cache = {}
        def compute_subtree_size(nid):
            if nid in subtree_cache:
                return subtree_cache[nid]
            size = 1
            for cid in self.swc_data[nid].get('children', []):
                size += compute_subtree_size(cid)
            subtree_cache[nid] = size
            return size

        for nid in self.swc_data:
            self.node_features[nid]['distance_from_root'] = dist_root.get(nid, 0)
            self.node_features[nid]['path_length_to_terminal'] = path_to_term.get(nid, 0)
            self.node_features[nid]['subtree_size'] = subtree_cache.get(nid, 0)

    def _normalize_coordinates(self, swc_data, mode='center_scale', origin_node_id: int = 0):

        if not swc_data:
            return swc_data

        # 提取所有坐标
        coords = np.array([
            [node['x'], node['y'], node['z']]
            for node in swc_data.values()
        ])

        # 应用中心化
        if mode in ['center', 'center_scale']:
            center = coords.mean(axis=0)
            coords = coords - center
        elif mode == 'root_scale':
            if origin_node_id is None or origin_node_id not in swc_data:
                raise ValueError("origin_node_id must be provided and exist in swc_data for 'root_scale' mode.")
            root = swc_data[origin_node_id]
            coords = coords - np.array([root['x'], root['y'], root['z']], dtype=np.float32)

        # 缩放
        if mode in ['scale', 'center_scale', 'root_scale']:
            norm = np.linalg.norm(coords, axis=1).max()
            if not np.isclose(norm, 0.0, atol=1e-6):
                coords /= norm
            else:
                warnings.warn("坐标范围接近零，跳过归一化缩放。")

        # 回写到 swc_data
        for node, pos in zip(swc_data.values(), coords):
            node['x'], node['y'], node['z'] = map(float, pos)

        return swc_data

    def _compute_strahler_order(self):
        root = self._get_root_id()
        strahler = {nid: 1 for nid in self.swc_data}
        stack = [(root, False)]

        while stack:
            nid, processed = stack.pop()
            if not processed:
                stack.append((nid, True))
                for cid in reversed(self.swc_data[nid].get('children', [])):
                    stack.append((cid, False))
            else:
                children = self.swc_data[nid].get('children', [])
                if not children:
                    strahler[nid] = 1
                else:
                    orders = [strahler[cid] for cid in children]
                    max_o = max(orders)
                    strahler[nid] = max_o + 1 if orders.count(max_o) >= 2 else max_o
                self.node_features[nid]['strahler_order'] = strahler[nid]



    def _get_root_id(self) -> int:
        for nid, node in self.swc_data.items():
            if node.get('parent') == -1:
                return nid
        raise ValueError("SWC 数据中未找到根节点")






# 示例使用
if __name__ == "__main__":
    # 示例SWC数据
    example_swc = {
        0: {'id': 0, 'type': 1, 'x': 375.2, 'y': 548.5, 'z': 19.3, 'radius': 4.8, 'parent': -1, 'children': [1, 2]},
        1: {'id': 1, 'type': 2, 'x': 380.1, 'y': 550.3, 'z': 20.1, 'radius': 3.2, 'parent': 0, 'children': [3, 4]},
        2: {'id': 2, 'type': 2, 'x': 370.5, 'y': 545.0, 'z': 18.9, 'radius': 3.5, 'parent': 0, 'children': [5, 6]},
        3: {'id': 3, 'type': 3, 'x': 385.0, 'y': 555.0, 'z': 21.0, 'radius': 2.1, 'parent': 1, 'children': []},
        4: {'id': 4, 'type': 3, 'x': 382.0, 'y': 548.0, 'z': 19.5, 'radius': 2.3, 'parent': 1, 'children': [7]},
        5: {'id': 5, 'type': 3, 'x': 368.0, 'y': 540.0, 'z': 17.5, 'radius': 2.8, 'parent': 2, 'children': []},
        6: {'id': 6, 'type': 3, 'x': 365.0, 'y': 550.0, 'z': 19.5, 'radius': 2.5, 'parent': 2, 'children': [8, 9]},
        7: {'id': 7, 'type': 4, 'x': 384.0, 'y': 545.0, 'z': 18.0, 'radius': 1.8, 'parent': 4, 'children': []},
        8: {'id': 8, 'type': 4, 'x': 363.0, 'y': 555.0, 'z': 20.5, 'radius': 1.9, 'parent': 6, 'children': []},
        9: {'id': 9, 'type': 4, 'x': 360.0, 'y': 545.0, 'z': 18.5, 'radius': 1.7, 'parent': 6, 'children': []}
    }


    from dataloader.io.read_swc import read_swc

    morphology = read_swc('../notebook/cell_types/specimen_485909730/reconstruction.swc')

    # 创建特征计算器实例
    calculator = NeuronFeatureCalculator(morphology.compartment_index)
    # calculator = NeuronNodeFeatureCalculator(example_swc)
    node_feature = calculator.node_features
    edge_feature = calculator.edge_features
    print(node_feature)
    # 获取特征结果
    # features = {
    #     'node_features': calculator.node_features,
    # }
    #
    # # 打印部分特征
    # print("=== 节点特征示例 ===")
    # print("\n根节点(0)特征:")
    # for k, v in features['node_features'][0].items():
    #     print(f"{k}: {v}")
    #
    # print("\n分支节点(1)特征:")
    # for k, v in features['node_features'][1].items():
    #     print(f"{k}: {v}")
    #
    # print("\n分支节点(13)特征:")
    # for k, v in features['node_features'][13].items():
    #     print(f"{k}: {v}")