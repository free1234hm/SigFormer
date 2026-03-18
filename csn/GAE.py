#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 10:35
# @Author  : Xiao Li
# @File    : main.py
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn import TransformerConv, InnerProductDecoder
from sklearn.metrics import accuracy_score, f1_score


output_dir = "./perturbation_debug"
os.makedirs(output_dir, exist_ok=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = TransformerConv(in_channels1, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.decoder_0 = InnerProductDecoder()
        self.decoder_1 = InnerProductDecoder()
        self.fc1 = torch.nn.Linear(in_channels2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def initialize_weights(self):
        self.conv1.reset_parameters()
        self.conv3.reset_parameters()

    def encode(self, x, train_edge_index):
        x = F.relu(self.conv1(x, train_edge_index))
        x = self.conv3(x, train_edge_index)
        return x

    def forward(self, x, train_edge_index_0, train_edge_index_1):
        z = self.encode(x, train_edge_index_1)

        # 保持原始 pooling 逻辑不变
        y1, _ = torch.max(z, dim=1)
        y1_1 = y1.unsqueeze(0)
        y2 = F.relu(self.fc1(y1_1))
        y = self.fc2(y2)

        adj_reconstructed_0 = self.decoder_0(z, train_edge_index_0)
        adj_reconstructed_1 = self.decoder_1(z, train_edge_index_1)
        return adj_reconstructed_0, adj_reconstructed_1, z, y


def filter_edges_by_weight(edge_index, edge_weight, filter_value):
    mask = edge_weight == filter_value
    filtered_edge_index = edge_index[:, mask]
    return filtered_edge_index, mask


def evaluate_model(model, data_list, device, criterion_0, criterion_1, criterion_2):
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

                recon_adj_0, recon_adj_1, z, y = model(
                    x,
                    train_edge_index_0.to(device),
                    train_edge_index_1.to(device)
                )

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

    accuracy = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    f1_macro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    f1_micro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='micro')
    f1_weighted = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='weighted')

    print(f'Test Loss: {total_loss:.8f}')
    print(
        f'Accuracy:\t{accuracy:.4f}\t'
        f'F1_score_macro:\t{f1_macro:.4f}\t'
        f'F1_score_micro:\t{f1_micro:.4f}\t'
        f'F1_score_weighted:\t{f1_weighted:.4f}'
    )
    return total_loss, accuracy, f1_macro, f1_micro, f1_weighted


def trimmed_mean(z_stack, trim_ratio: float = 0.1):
    sorted_z, _ = torch.sort(z_stack, dim=0)
    N = z_stack.size(0)
    lower_idx = int(N * trim_ratio)
    upper_idx = int(N * (1 - trim_ratio))
    trimmed_z = sorted_z[lower_idx:upper_idx]
    trimmed_mean_z = torch.mean(trimmed_z, dim=0)
    return trimmed_mean_z


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def get_positive_edge_index(edge_index, edge_attr):
    mask = edge_attr == 1
    return edge_index[:, mask]


def gene_has_positive_edge(edge_index_1, gene_idx):
    return bool(((edge_index_1[0] == gene_idx) | (edge_index_1[1] == gene_idx)).any())


def knockout_graph(data, knockout_gene_idx):
    """
    敲除：删边 + 节点特征置零
    knockout_gene_idx: int 或可迭代对象
    """
    if isinstance(knockout_gene_idx, (int, np.integer)):
        ko_idx = torch.tensor([int(knockout_gene_idx)], dtype=torch.long)
    else:
        ko_idx = torch.as_tensor(list(knockout_gene_idx), dtype=torch.long)

    x_ko = data.x.clone()
    x_ko[ko_idx] = 0.0

    edge_attr_ko = data.edge_attr.clone()
    src = data.edge_index[0]
    dst = data.edge_index[1]

    edge_mask = torch.isin(src, ko_idx) | torch.isin(dst, ko_idx)
    edge_attr_ko[edge_mask] = 0.0

    return x_ko, edge_attr_ko


def encode_single_graph(model, x, edge_index_1, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(x.to(device), edge_index_1.to(device))
    return z.cpu()


def compute_knockout_distances_for_cell(model, data, knockout_gene_idx, device):
    """
    对单个细胞，计算敲除前后每个基因 embedding 的 L2 距离
    如果待敲除基因在当前细胞网络中没有任何正边，则返回 None
    """
    edge_index_1 = get_positive_edge_index(data.edge_index, data.edge_attr)

    if edge_index_1.shape[1] <= 1:
        # print(f"边不足")
        return None

    if not gene_has_positive_edge(edge_index_1, knockout_gene_idx):
        return None

    z = encode_single_graph(model, data.x, edge_index_1, device)

    x_ko, edge_attr_ko = knockout_graph(data, knockout_gene_idx)
    edge_index_1_ko = get_positive_edge_index(data.edge_index, edge_attr_ko)

    if edge_index_1_ko.shape[1] <= 1:
        return None

    z_ko = encode_single_graph(model, x_ko, edge_index_1_ko, device)
    dist = torch.norm(z - z_ko, p=2, dim=1)  # [num_genes]
    return dist


def aggregate_distances_by_label(distance_dict, agg='median'):
    aggregated = {}
    for label, dist_list in distance_dict.items():
        if len(dist_list) == 0:
            continue
        mat = torch.stack(dist_list, dim=0)  # [n_cells, n_genes]
        if agg == 'mean':
            aggregated[label] = mat.mean(dim=0).cpu().numpy()
        else:
            aggregated[label] = mat.median(dim=0).values.cpu().numpy()
    return aggregated


def collect_observed_knockout_scores_single_gene(model, data_list, cell_idx,
                                                 device, knockout_gene_idx, only_correct=True):
    """
    收集真实单基因 knockout 下，各 cell type 的扰动分数
    perturbation阶段只要求 edge_index_1 非空，不再要求 edge_index_0 足够多
    """
    obs_distance_dict = {}

    with torch.no_grad():
        for idx in cell_idx:
            data = data_list[idx]
            edge_index_0, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            edge_index_1, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if edge_index_1.shape[1] <= 1:
                # print(f"边不足")
                continue

            label = int(data.y.cpu().item())

            if only_correct:
                x = data.x.to(device)
                _, _, _, y = model(x, edge_index_0.to(device), edge_index_1.to(device))
                pred = int(y.argmax(dim=1).cpu().item())
                if pred != label:
                    # print(f"细胞类型预测错误")
                    continue

            dist = compute_knockout_distances_for_cell(model, data, knockout_gene_idx, device)
            if dist is None:
                continue

            if label not in obs_distance_dict:
                obs_distance_dict[label] = []
            obs_distance_dict[label].append(dist)
    '''
    for label, dists in obs_distance_dict.items():
        if len(dists) == 0:
            continue
        obs_mat = torch.stack(dists, dim=0).cpu().numpy()  # [n_cells, num_genes]
        obs_df = pd.DataFrame(obs_mat)
        obs_df.to_csv(os.path.join(output_dir, f"{label}_observed.csv"), index=False)
    '''

    return obs_distance_dict


def collect_random_knockout_background(
    model,
    data_list,
    cell_idx,
    device,
    num_random_knockouts=500,
    agg='median',
    seed=42,
    only_correct=True
):
    """
    只构建一次共享 random knockout 背景。
    因为每次都是单基因随机敲除，所以后续所有 true knockout 都共用这个背景。
    返回:
        random_bg[label] = np.ndarray [B, num_genes]
    """
    rng = np.random.default_rng(seed)
    num_genes = data_list[0].x.shape[0]
    candidate_genes = np.arange(num_genes, dtype=int)

    random_bg = {}

    for b in range(num_random_knockouts):
        rand_ko = int(rng.choice(candidate_genes, size=1, replace=False)[0])

        dist_dict_b = {}
        with torch.no_grad():
            for idx in cell_idx:
                data = data_list[idx]
                edge_index_0, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
                edge_index_1, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

                if edge_index_1.shape[1] <= 1:
                    # print(f"边不足")
                    continue

                label = int(data.y.cpu().item())

                if only_correct:
                    x = data.x.to(device)
                    _, _, _, y = model(x, edge_index_0.to(device), edge_index_1.to(device))
                    pred = int(y.argmax(dim=1).cpu().item())
                    if pred != label:
                        # print(f"细胞类型预测错误")
                        continue

                dist = compute_knockout_distances_for_cell(model, data, rand_ko, device)
                if dist is None:
                    continue

                if label not in dist_dict_b:
                    dist_dict_b[label] = []
                dist_dict_b[label].append(dist)

        agg_b = aggregate_distances_by_label(dist_dict_b, agg=agg)

        for label, score_vec in agg_b.items():
            if label not in random_bg:
                random_bg[label] = []
            random_bg[label].append(score_vec)

        print("\r", end="")
        print(f"Random knockout background: {b + 1}/{num_random_knockouts}", end="")
    print()

    for label in random_bg:
        random_bg[label] = np.stack(random_bg[label], axis=0)  # [B, num_genes]
    '''
    for label in random_bg:
        random_df = pd.DataFrame(random_bg[label])
        random_df.to_csv(os.path.join(output_dir, f"{label}_random.csv"), index=False)
    '''

    return random_bg


def get_significant_gene_idx_from_background(obs_scores, random_bg, cut_off=0.05):
    """
    只返回每个 label 下显著扰动基因 index 的 list
    """
    result = {}
    for label, obs in obs_scores.items():
        if label not in random_bg:
            continue

        bg = random_bg[label]  # [B, num_genes]
        pvals = (1.0 + np.sum(bg >= obs[None, :], axis=0)) / (bg.shape[0] + 1.0)
        # fdr = benjamini_hochberg(pvals)
        significant_idx = np.where(pvals < cut_off)[0].tolist()
        result[label] = significant_idx
    return result


def graph_processing(
    data_list,
    cell_idx,
    gene_names,
    cell_labels,
    embedding_dim: int = 32,
    out_dim: int = None,
    num_epochs: int = 50,
    learning_rate=0.001,
    min_cell_count: int = 5,
    seed: int = 42,

    # perturbation 参数
    knockout_gene_idx=None,
    num_random_knockouts: int = 500,
    perturbation_agg: str = 'mean',
    perturbation_fdr_alpha: float = 0.05,
    only_correct_cells_for_perturbation: bool = True
):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphAutoencoder(
        in_channels1=data_list[0].num_node_features,
        in_channels2=data_list[0].num_nodes,
        hidden_channels=embedding_dim,
        out_channels=out_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_0, criterion_1, criterion_2 = torch.nn.BCELoss(), torch.nn.BCELoss(), torch.nn.CrossEntropyLoss()

    min_loss = float('inf')
    patience = 5
    count_no_improve = 0

    # =========================
    # 保留原始训练部分
    # 训练阶段仍保留 edge_index_0 / edge_index_1 双重过滤
    # =========================
    for epoch in range(num_epochs):
        model.train()
        print('epoch: %d' % epoch)

        for data in data_list:
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                optimizer.zero_grad()

                x = data.x.to(device)
                recon_adj_0, recon_adj_1, z, y = model(
                    x,
                    train_edge_index_0.to(device),
                    train_edge_index_1.to(device)
                )

                recon_adj_0 = recon_adj_0.squeeze()
                recon_adj_1 = recon_adj_1.squeeze()
                edge_attr_0 = data.edge_attr[mask_0]
                edge_attr_1 = data.edge_attr[mask_1]

                data.y = data.y.view(-1)
                loss_0 = criterion_0(recon_adj_0, edge_attr_0.to(device))
                loss_1 = criterion_1(recon_adj_1, edge_attr_1.to(device))
                loss_2 = criterion_2(y, data.y.to(device))
                loss = 0.5 * loss_0 + 0.5 * loss_1 + loss_2

                loss.backward()
                optimizer.step()

        total_loss, accuracy, f1_macro, f1_micro, f1_weighted = evaluate_model(
            model, data_list, device, criterion_0, criterion_1, criterion_2
        )

        if total_loss < min_loss:
            count_no_improve = 0
            min_loss = total_loss
        else:
            count_no_improve += 1

        if count_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no loss improvement for {patience} epochs)")
            break

    # =========================
    # 原始 eval：重构 cell-type-specific network
    # =========================
    model.eval()

    z_dict = {}
    y_true, y_pred = [], []

    with torch.no_grad():
        for idx in cell_idx:
            data = data_list[idx]
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                label = int(data.y.cpu().item())
                y_true.append(label)

                x = data.x.to(device)
                recon_adj_0, recon_adj_1, z, y = model(
                    x,
                    train_edge_index_0.to(device),
                    train_edge_index_1.to(device)
                )

                label_pred = int(y.argmax(dim=1).cpu().item())
                y_pred.append(label_pred)

                latent_adj = z.cpu()
                if label == label_pred:
                    if label not in z_dict:
                        z_dict[label] = []
                    z_dict[label].append(latent_adj)

    edge_index = data_list[0].edge_index
    adj_dict_mean = {}
    for label, z_list in z_dict.items():
        if len(z_list) >= min_cell_count:
            z_stack = torch.stack(z_list)
            joint_z_mean = trimmed_mean(z_stack)

            decoder1 = InnerProductDecoder()
            recon_adj_mean = decoder1(joint_z_mean, edge_index)
            recon_adj_mean = recon_adj_mean.cpu().numpy()

            row, col = edge_index.cpu().numpy()
            adj_matrix_mean = coo_matrix(
                (recon_adj_mean, (row, col)),
                shape=(data_list[0].x.shape[0], data_list[0].x.shape[0])
            )
            adj_dict_mean[label] = adj_matrix_mean

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy:\t{accuracy:.4f}\tF1_score_macro:\t{f1_macro:.4f}\t'
          f'F1_score_micro:\t{f1_micro:.4f}\tF1_score_weighted:\t{f1_weighted:.4f}')

    # =========================
    # single-cell resolved perturbation analysis
    # 1. 先统一构建一次共享 random knockout 背景
    # 2. 再循环每个 true knockout
    # 3. perturbation_results[label][ko_gene] = significant_gene_idx_list
    # =========================
    perturbation_results = {}

    if knockout_gene_idx is not None and len(knockout_gene_idx) > 0:
        print('Start building random knockout background...')
        shared_random_bg = collect_random_knockout_background(
            model=model,
            cell_idx=cell_idx,
            data_list=data_list,
            device=device,
            num_random_knockouts=num_random_knockouts,
            agg=perturbation_agg,
            seed=seed,
            only_correct=only_correct_cells_for_perturbation
        )

        for ko_gene in knockout_gene_idx:
            print("\r", end="")
            print(f"Knockout analysis for receptor: {gene_names[ko_gene]}", end="")

            obs_distance_dict = collect_observed_knockout_scores_single_gene(
                model=model,
                data_list=data_list,
                cell_idx=cell_idx,
                device=device,
                knockout_gene_idx=ko_gene,
                only_correct=only_correct_cells_for_perturbation
            )
            obs_scores = aggregate_distances_by_label(obs_distance_dict, agg=perturbation_agg)

            sig_result = get_significant_gene_idx_from_background(
                obs_scores=obs_scores,
                random_bg=shared_random_bg,
                cut_off=perturbation_fdr_alpha
            )

            for label, significant_gene_idx in sig_result.items():
                cell_type = cell_labels[label]
                if cell_type not in perturbation_results:
                    perturbation_results[cell_type] = {}
                perturbation_results[cell_type][gene_names[ko_gene]] = [gene_names[i] for i in significant_gene_idx]
        print()

    return accuracy, adj_dict_mean, perturbation_results
