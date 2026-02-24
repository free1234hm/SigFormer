#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 10:35
# @Author  : Xiao Li
# @File    : main.py
import os
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.sparse import coo_matrix, diags
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, InnerProductDecoder, GATConv, GATv2Conv, TransformerConv, global_max_pool, \
    global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        # self.conv1 = GCNConv(in_channels1, out_channels)
        # self.conv3 = GCNConv(out_channels, out_channels)
        self.conv1 = TransformerConv(in_channels1, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.decoder_0 = InnerProductDecoder()
        self.decoder_1 = InnerProductDecoder()
        self.fc1 = torch.nn.Linear(in_channels2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

        # 初始化权重
        # self.initialize_weights()

    def initialize_weights(self):
        self.conv1.reset_parameters()
        self.conv3.reset_parameters()

    def encode(self, x, train_edge_index):
        x = F.relu(self.conv1(x, train_edge_index))
        x = self.conv3(x, train_edge_index)
        return x

    def forward(self, x, train_edge_index_0, train_edge_index_1):
        z = self.encode(x, train_edge_index_1)

        # batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        # 全局池化，获取图级表示
        y1, _ = torch.max(z, dim=1)
        # y1_1 = global_mean_pool(z, batch)
        # 分类层
        y1_1 = y1.unsqueeze(0)
        # print(y1.shape, y1_1.shape)
        y2 = F.relu(self.fc1(y1_1))  # 添加一个维度使其适配 fc1
        y = self.fc2(y2)

        adj_reconstructed_0 = self.decoder_0(z, train_edge_index_0)
        adj_reconstructed_1 = self.decoder_1(z, train_edge_index_1)
        return adj_reconstructed_0, adj_reconstructed_1, z, y


def filter_edges_by_weight(edge_index, edge_weight, filter):
    # 获取权重为1的边的布尔索引
    mask = edge_weight == filter
    # 使用布尔索引筛选出边的索引和权重
    filtered_edge_index = edge_index[:, mask]
    return filtered_edge_index, mask


def evaluate_model(model, data_list, device, criterion_0, criterion_1, criterion_2):
    """评估模型的性能，计算准确率和F1 score"""
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for data in data_list:
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)
            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                x = data.x.to(device)
                y_true.append(data.y.cpu().numpy())

                recon_adj_0, recon_adj_1, z, y = model(x, train_edge_index_0.to(device), train_edge_index_1.to(device))

                recon_adj_0 = recon_adj_0.squeeze()
                recon_adj_1 = recon_adj_1.squeeze()
                edge_attr_0 = data.edge_attr[mask_0]
                edge_attr_1 = data.edge_attr[mask_1]

                loss = (
                        0.5 * criterion_0(recon_adj_0, edge_attr_0.to(device)) +
                        0.5 * criterion_1(recon_adj_1, edge_attr_1.to(device)) +
                        criterion_2(y, data.y.to(device))
                )
                total_loss += loss.item()

                y_pred.append(y.argmax(dim=1).cpu().numpy())

    # 计算准确率和F1 score
    accuracy = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    f1_macro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    f1_micro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='micro')
    f1_weighted = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='weighted')

    print(f'Test Loss: {total_loss:.8f}')
    print(f'Accuracy:\t{accuracy:.4f}\tF1_score_macro:\t{f1_macro:.4f}\tF1_score_micro:\t{f1_micro:.4f}\tF1_score_weighted:\t{f1_weighted:.4f}')
    return total_loss, accuracy, f1_macro, f1_micro, f1_weighted


def trimmed_mean(z_stack, trim_ratio: float = 0.1):
    sorted_z, _ = torch.sort(z_stack, dim=0)

    # Determine the trimming indices
    N = z_stack.size(0)  # Number of matrices
    lower_idx = int(N * trim_ratio)  # Index to start trimming from the bottom
    upper_idx = int(N * (1 - trim_ratio))  # Index to stop trimming from the top

    trimmed_z = sorted_z[lower_idx:upper_idx]  # Shape: (trimmed_N, 100, 3)

    # Compute the mean of the trimmed tensor
    trimmed_mean_z = torch.mean(trimmed_z, dim=0)  # Shape: (100, 3)
    return trimmed_mean_z


def graph_processing(data_list,
                     cell_names,
                     embedding_dim: int = 32,
                     out_dim: int = None,
                     num_epochs: int = 50,
                     learning_rate = 0.001,
                     min_cell_count: int = 5,
                     seed: int = 42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型、损失函数和优化器
    model = GraphAutoencoder(in_channels1=data_list[0].num_node_features, in_channels2=data_list[0].num_nodes,
                             hidden_channels=embedding_dim, out_channels=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_0, criterion_1, criterion_2 = torch.nn.BCELoss(), torch.nn.BCELoss(), torch.nn.CrossEntropyLoss()

    pre_acc = -float('inf')  # 当前最佳 accuracy
    min_delta = 1e-3  # 判断“是否提升”的最小变化（可调）
    patience = 5  # 允许的连续无改善轮数
    count_no_improve = 0  # 计数器
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_0, total_loss_1, total_loss_2 = 0, 0, 0
        model.train()
        print('epoch: %d' % epoch)
        for data in data_list:
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)
            # print(f"Shape of recon_adj_1: {train_edge_index_1.shape[0]} Shape of edge_attr_1: {train_edge_index_1.shape[1]}")
            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                optimizer.zero_grad()

                x = data.x.to(device)
                recon_adj_0, recon_adj_1, z, y = model(x, train_edge_index_0.to(device), train_edge_index_1.to(device))

                recon_adj_0 = recon_adj_0.squeeze()
                recon_adj_1 = recon_adj_1.squeeze()
                edge_attr_0 = data.edge_attr[mask_0]
                edge_attr_1 = data.edge_attr[mask_1]

                data.y = data.y.view(-1)  # 将 data.y 的形状调整为 (batch_size,)
                loss_0 = criterion_0(recon_adj_0, edge_attr_0.to(device))
                loss_1 = criterion_1(recon_adj_1, edge_attr_1.to(device))
                loss_2 = criterion_2(y, data.y.to(device))
                loss = 0.5 * loss_0 + 0.5 * loss_1 + loss_2
                # loss = loss_2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss_0 += loss_0.item()
                total_loss_1 += loss_1.item()
                total_loss_2 += loss_2.item()

        # print(f'current loss: {total_loss:.8f}', f'current loss_0: {total_loss_0:.8f}',
        #       f'current loss_1: {total_loss_1:.8f}', f'current loss_2: {total_loss_2:.8f}')
        # if total_loss < min_loss:
        # print('current_loss:{} < min_loss:{}'.format(total_loss, min_loss))
        # print('===========>save best model!')
        # min_loss = total_loss

        total_loss, accuracy, f1_macro, f1_micro, f1_weighted = evaluate_model(model, data_list, device, criterion_0,
                                                                               criterion_1, criterion_2)

        # ---------------------------------------------
        # 检查是否有明显提升
        # ---------------------------------------------
        if f1_weighted > pre_acc + min_delta:
            count_no_improve = 0
        else:
            count_no_improve += 1

        pre_acc = f1_weighted
        # ---------------------------------------------
        # 早停判断
        # ---------------------------------------------
        if count_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no acc improvement for {patience} epochs)")
            break

    model.eval()

    z_dict = {}
    rescon_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in range(len(data_list)):
            data = data_list[idx]
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)
            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                label = data.y.cpu().item()
                y_true.append(label)
                x = data.x.to(device)
                recon_adj_0, recon_adj_1, z, y = model(x, train_edge_index_0.to(device), train_edge_index_1.to(device))

                recon_adj_0 = recon_adj_0.cpu().numpy()
                recon_adj_1 = recon_adj_1.cpu().numpy()
                label_pred = y.argmax(dim=1).cpu().item()
                y_pred.append(label_pred)

                cell_loss = 0
                for val in recon_adj_0:
                    cell_loss += abs(val)
                for val in recon_adj_1:
                    cell_loss += abs(1 - val)
                rescon_loss += cell_loss

                latent_adj = z.cpu()
                # z_list.append(latent_adj)
                if label == label_pred:
                    if label not in z_dict:
                        z_dict[label] = []
                    z_dict[label].append(latent_adj)

                '''
                recon_dir = f'./result/recon_network/{label}'
                os.makedirs(recon_dir, exist_ok=True)

                adj_file = cell_names[idx]
                row_0, col_0 = train_edge_index_0.cpu().numpy()
                row_1, col_1 = train_edge_index_1.cpu().numpy()
                recon_output_file = os.path.join(recon_dir, f'{adj_file}_reconstructed.txt')
                with open(recon_output_file, 'w') as f:
                    f.write(f"{data.x.shape[0]}\t{data.x.shape[0]}\n")
                    for r, c, w in zip(row_1, col_1, recon_adj_1):
                        f.write(f"{r}\t{c}\t{w}\n")
                    for r, c, w in zip(row_0, col_0, recon_adj_0):
                        f.write(f"{r}\t{c}\t{w}\n")
                '''

    edge_index = data_list[0].edge_index
    adj_dict_mean = {}
    for label, z_list in z_dict.items():
        if len(z_list) >= min_cell_count:
            z_stack = torch.stack(z_list)
            joint_z_median = torch.median(z_stack, dim=0).values  # Shape: [num_nodes, embedding_dim]
            joint_z_mean = trimmed_mean(z_stack)

            # decoder0 = InnerProductDecoder()
            # recon_adj_median = decoder0(joint_z_median, edge_index)
            # recon_adj_median = recon_adj_median.cpu().numpy()

            decoder1 = InnerProductDecoder()
            recon_adj_mean = decoder1(joint_z_mean, edge_index)
            recon_adj_mean = recon_adj_mean.cpu().numpy()

            row, col = edge_index.cpu().numpy()
            adj_matrix_mean = coo_matrix((recon_adj_mean, (row, col)),
                                         shape=(data_list[0].x.shape[0], data_list[0].x.shape[0]))
            adj_dict_mean[label] = adj_matrix_mean

    '''
    row, col = np.where(recon_adj_np > 0)
    values = recon_adj_np[row, col]  # Corresponding values
    data_to_save = np.column_stack((row, col, values))
    np.savetxt("./result/recon_network.txt", data_to_save, fmt=["%d", "%d", "%.6f"], delimiter="\t",
               header=f"{recon_adj_np.shape[0]}\t{recon_adj_np.shape[1]}", comments="")
    '''

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 使用 average='macro' 计算 F1 分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # 使用 average='micro' 计算 F1 分数
    f1_micro = f1_score(y_true, y_pred, average='micro')
    # 使用 average='weighted' 计算 F1 分数
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(
        f'Accuracy:\t{accuracy:.4f}\tF1_score_macro:\t{f1_macro:.4f}\tF1_score_micro:\t{f1_micro:.4f}\tF1_score_weighted:\t{f1_weighted:.4f}')

    return accuracy, adj_dict_mean


def graph_processing_spatial(data_list,
                             cell_names,
                             cell_neighbors,
                             embedding_dim: int = 32,
                             out_dim: int = None,
                             num_epochs: int = 50,
                             learning_rate = 0.001,
                             min_cell_count: int = 5,
                             seed: int = 42):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型、损失函数和优化器
    model = GraphAutoencoder(in_channels1=data_list[0].num_node_features, in_channels2=data_list[0].num_nodes,
                             hidden_channels=embedding_dim, out_channels=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_0, criterion_1, criterion_2 = torch.nn.BCELoss(), torch.nn.BCELoss(), torch.nn.CrossEntropyLoss()

    pre_acc = -float('inf')  # 当前最佳 accuracy
    min_delta = 1e-3  # 判断“是否提升”的最小变化（可调）
    patience = 5  # 允许的连续无改善轮数
    count_no_improve = 0  # 计数器
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_0, total_loss_1, total_loss_2 = 0, 0, 0
        model.train()
        print('epoch: %d' % epoch)
        for data in data_list:
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)
            # print(f"Shape of recon_adj_1: {train_edge_index_1.shape[0]} Shape of edge_attr_1: {train_edge_index_1.shape[1]}")
            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                optimizer.zero_grad()

                x = data.x.to(device)
                recon_adj_0, recon_adj_1, z, y = model(x, train_edge_index_0.to(device), train_edge_index_1.to(device))

                recon_adj_0 = recon_adj_0.squeeze()
                recon_adj_1 = recon_adj_1.squeeze()
                edge_attr_0 = data.edge_attr[mask_0]
                edge_attr_1 = data.edge_attr[mask_1]

                data.y = data.y.view(-1)  # 将 data.y 的形状调整为 (batch_size,)
                loss_0 = criterion_0(recon_adj_0, edge_attr_0.to(device))
                loss_1 = criterion_1(recon_adj_1, edge_attr_1.to(device))
                loss_2 = criterion_2(y, data.y.to(device))
                loss = 0.5 * loss_0 + 0.5 * loss_1 + loss_2
                # loss = loss_2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss_0 += loss_0.item()
                total_loss_1 += loss_1.item()
                total_loss_2 += loss_2.item()

        total_loss, accuracy, f1_macro, f1_micro, f1_weighted = evaluate_model(model, data_list, device, criterion_0,
                                                                               criterion_1, criterion_2)

        # ---------------------------------------------
        # 检查是否有明显提升
        # ---------------------------------------------
        if f1_weighted > pre_acc + min_delta:
            count_no_improve = 0
        else:
            count_no_improve += 1

        pre_acc = f1_weighted

        # ---------------------------------------------
        # 早停判断
        # ---------------------------------------------
        if count_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no acc improvement for {patience} epochs)")
            break


    model.eval()

    z_dict = {}
    rescon_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in cell_neighbors:
            data = data_list[idx]
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)
            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                label = data.y.cpu().item()
                y_true.append(label)
                x = data.x.to(device)
                recon_adj_0, recon_adj_1, z, y = model(x, train_edge_index_0.to(device), train_edge_index_1.to(device))

                recon_adj_0 = recon_adj_0.cpu().numpy()
                recon_adj_1 = recon_adj_1.cpu().numpy()
                label_pred = y.argmax(dim=1).cpu().item()
                y_pred.append(label_pred)

                cell_loss = 0
                for val in recon_adj_0:
                    cell_loss += abs(val)
                for val in recon_adj_1:
                    cell_loss += abs(1 - val)
                rescon_loss += cell_loss

                latent_adj = z.cpu()
                # z_list.append(latent_adj)
                if label == label_pred:
                    if label not in z_dict:
                        z_dict[label] = []
                    z_dict[label].append(latent_adj)

    edge_index = data_list[0].edge_index
    adj_dict_mean = {}
    for label, z_list in z_dict.items():
        if len(z_list) >= min_cell_count:
            z_stack = torch.stack(z_list)
            joint_z_median = torch.median(z_stack, dim=0).values  # Shape: [num_nodes, embedding_dim]
            joint_z_mean = trimmed_mean(z_stack)

            # decoder0 = InnerProductDecoder()
            # recon_adj_median = decoder0(joint_z_median, edge_index)
            # recon_adj_median = recon_adj_median.cpu().numpy()

            decoder1 = InnerProductDecoder()
            recon_adj_mean = decoder1(joint_z_mean, edge_index)
            recon_adj_mean = recon_adj_mean.cpu().numpy()

            row, col = edge_index.cpu().numpy()
            adj_matrix_mean = coo_matrix((recon_adj_mean, (row, col)),
                                         shape=(data_list[0].x.shape[0], data_list[0].x.shape[0]))
            adj_dict_mean[label] = adj_matrix_mean

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 使用 average='macro' 计算 F1 分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # 使用 average='micro' 计算 F1 分数
    f1_micro = f1_score(y_true, y_pred, average='micro')
    # 使用 average='weighted' 计算 F1 分数
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(
        f'Accuracy:\t{accuracy:.4f}\tF1_score_macro:\t{f1_macro:.4f}\tF1_score_micro:\t{f1_micro:.4f}\tF1_score_weighted:\t{f1_weighted:.4f}')

    return accuracy, adj_dict_mean
