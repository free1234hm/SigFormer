#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 10:35
# @Author  : Xiao Li
# @File    : main.py
import os
import random
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn import TransformerConv
from sklearn.metrics import accuracy_score, f1_score


# output_dir = "./perturbation_debug"
# os.makedirs(output_dir, exist_ok=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DirectedRelationDecoder(torch.nn.Module):
    """Add a learnable directed, relation-aware residual to inner-product logits."""

    def __init__(
        self,
        latent_channels,
        num_relations,
        initial_residual_scale=0.1,
        max_residual_adjustment=0.5
    ):
        super().__init__()
        if not 0 < initial_residual_scale < 1:
            raise ValueError("initial_residual_scale must be between 0 and 1.")
        if max_residual_adjustment <= 0:
            raise ValueError("max_residual_adjustment must be positive.")

        self.latent_channels = latent_channels
        self.num_relations = num_relations
        self.initial_residual_scale = initial_residual_scale
        self.max_residual_adjustment = max_residual_adjustment
        self.source_projection = torch.nn.Linear(latent_channels, latent_channels, bias=False)
        self.target_projection = torch.nn.Linear(latent_channels, latent_channels, bias=False)
        self.relation_projection = torch.nn.Linear(num_relations, latent_channels, bias=False)
        self.relation_bias = torch.nn.Linear(num_relations, 1)
        self.residual_logit = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.source_projection.reset_parameters()
        self.target_projection.reset_parameters()
        torch.nn.init.zeros_(self.relation_projection.weight)
        torch.nn.init.zeros_(self.relation_bias.weight)
        torch.nn.init.zeros_(self.relation_bias.bias)
        initial_logit = math.log(
            self.initial_residual_scale / (1.0 - self.initial_residual_scale)
        )
        with torch.no_grad():
            self.residual_logit.fill_(initial_logit)

    @property
    def residual_scale(self):
        return torch.sigmoid(self.residual_logit)

    def _compute_residual_logits(
        self,
        projected_source_z,
        projected_target_z,
        edge_index,
        relation_weights
    ):
        source = projected_source_z[edge_index[0]]
        target = projected_target_z[edge_index[1]]
        relation_gate = 1.0 + torch.tanh(self.relation_projection(relation_weights))
        logits = (source * target * relation_gate).sum(dim=1) / math.sqrt(self.latent_channels)
        return logits + self.relation_bias(relation_weights).squeeze(1)

    def project_nodes(self, z):
        return self.source_projection(z), self.target_projection(z)

    def compute_logit_components(
        self,
        z,
        edge_index,
        edge_relation_attr=None,
        edge_symmetric_mask=None,
        projected_source_z=None,
        projected_target_z=None
    ):
        num_edges = edge_index.shape[1]
        if edge_relation_attr is None:
            edge_relation_attr = torch.ones(
                (num_edges, self.num_relations),
                dtype=z.dtype,
                device=z.device
            )
        else:
            edge_relation_attr = edge_relation_attr.to(device=z.device, dtype=z.dtype)

        if edge_relation_attr.ndim == 1:
            edge_relation_attr = edge_relation_attr.unsqueeze(1)
        if edge_relation_attr.shape != (num_edges, self.num_relations):
            raise ValueError(
                "edge_relation_attr must have shape "
                f"({num_edges}, {self.num_relations}), got {tuple(edge_relation_attr.shape)}."
            )

        relation_weights = edge_relation_attr / edge_relation_attr.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1.0)
        source_z = z[edge_index[0]]
        target_z = z[edge_index[1]]
        inner_product_logits = (source_z * target_z).sum(dim=1)
        if projected_source_z is None or projected_target_z is None:
            projected_source_z, projected_target_z = self.project_nodes(z)
        residual_logits = self._compute_residual_logits(
            projected_source_z,
            projected_target_z,
            edge_index,
            relation_weights
        )

        if edge_symmetric_mask is not None:
            edge_symmetric_mask = edge_symmetric_mask.to(device=z.device, dtype=torch.bool)
            if edge_symmetric_mask.shape != (num_edges,):
                raise ValueError(
                    "edge_symmetric_mask must have shape "
                    f"({num_edges},), got {tuple(edge_symmetric_mask.shape)}."
                )
            if edge_symmetric_mask.any():
                symmetric_edges = edge_index[:, edge_symmetric_mask]
                reverse_edges = symmetric_edges.flip(0)
                reverse_residual_logits = self._compute_residual_logits(
                    projected_source_z,
                    projected_target_z,
                    reverse_edges,
                    relation_weights[edge_symmetric_mask]
                )
                symmetric_residual_logits = 0.5 * (
                    residual_logits[edge_symmetric_mask] + reverse_residual_logits
                )
                residual_logits = residual_logits.clone()
                residual_logits[edge_symmetric_mask] = symmetric_residual_logits

        residual_adjustment = (
            self.max_residual_adjustment
            * torch.tanh(
                self.residual_scale
                * residual_logits
                / self.max_residual_adjustment
            )
        )
        logits = inner_product_logits + residual_adjustment
        return logits, inner_product_logits, residual_adjustment

    def forward(
        self,
        z,
        edge_index,
        edge_relation_attr=None,
        edge_symmetric_mask=None,
        sigmoid=True,
        projected_source_z=None,
        projected_target_z=None
    ):
        logits, _, _ = self.compute_logit_components(
            z,
            edge_index,
            edge_relation_attr,
            edge_symmetric_mask,
            projected_source_z,
            projected_target_z
        )
        return torch.sigmoid(logits) if sigmoid else logits


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_channels, out_channels, num_relations=1):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = TransformerConv(in_channels1, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.decoder = DirectedRelationDecoder(hidden_channels, num_relations)
        self.fc1 = torch.nn.Linear(in_channels2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def initialize_weights(self):
        self.conv1.reset_parameters()
        self.conv3.reset_parameters()
        self.decoder.reset_parameters()

    def encode(self, x, train_edge_index):
        x = F.relu(self.conv1(x, train_edge_index))
        x = self.conv3(x, train_edge_index)
        return x

    def decode(
        self,
        z,
        edge_index,
        edge_relation_attr=None,
        edge_symmetric_mask=None,
        sigmoid=True,
        projected_source_z=None,
        projected_target_z=None
    ):
        return self.decoder(
            z,
            edge_index,
            edge_relation_attr,
            edge_symmetric_mask,
            sigmoid=sigmoid,
            projected_source_z=projected_source_z,
            projected_target_z=projected_target_z
        )

    def forward(
        self,
        x,
        train_edge_index_0,
        train_edge_index_1,
        edge_relation_attr_0=None,
        edge_relation_attr_1=None,
        edge_symmetric_mask_0=None,
        edge_symmetric_mask_1=None
    ):
        z = self.encode(x, train_edge_index_1)

        # Keep the original pooling logic unchanged.
        y1, _ = torch.max(z, dim=1)
        y1_1 = y1.unsqueeze(0)
        y2 = F.relu(self.fc1(y1_1))
        y = self.fc2(y2)
        projected_source_z, projected_target_z = self.decoder.project_nodes(z)

        adj_reconstructed_0 = self.decode(
            z,
            train_edge_index_0,
            edge_relation_attr_0,
            edge_symmetric_mask_0,
            sigmoid=False,
            projected_source_z=projected_source_z,
            projected_target_z=projected_target_z
        )
        adj_reconstructed_1 = self.decode(
            z,
            train_edge_index_1,
            edge_relation_attr_1,
            edge_symmetric_mask_1,
            sigmoid=False,
            projected_source_z=projected_source_z,
            projected_target_z=projected_target_z
        )
        return adj_reconstructed_0, adj_reconstructed_1, z, y


def filter_edges_by_weight(edge_index, edge_weight, filter_value):
    mask = edge_weight == filter_value
    filtered_edge_index = edge_index[:, mask]
    return filtered_edge_index, mask


def get_edge_relation_attr(data, mask=None):
    edge_relation_attr = getattr(data, 'edge_relation_attr', None)
    if edge_relation_attr is None:
        edge_relation_attr = torch.ones((data.edge_index.shape[1], 1), dtype=torch.float)
    if edge_relation_attr.ndim == 1:
        edge_relation_attr = edge_relation_attr.unsqueeze(1)
    if mask is not None:
        edge_relation_attr = edge_relation_attr[mask]
    return edge_relation_attr


def get_edge_symmetric_mask(data, mask=None):
    edge_symmetric_mask = getattr(data, 'edge_symmetric_mask', None)
    if edge_symmetric_mask is None:
        edge_symmetric_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    if mask is not None:
        edge_symmetric_mask = edge_symmetric_mask[mask]
    return edge_symmetric_mask


def summarize_decoder_adjustment(
    inner_product_logits,
    residual_adjustment,
    max_samples=100000
):
    """Summarize the actual residual contribution without retaining large tensors."""
    num_edges = residual_adjustment.numel()
    if num_edges == 0:
        return None

    if num_edges > max_samples:
        sample_indices = torch.linspace(
            0,
            num_edges - 1,
            steps=max_samples,
            device=residual_adjustment.device
        ).long()
        inner_product_logits = inner_product_logits[sample_indices]
        residual_adjustment = residual_adjustment[sample_indices]

    abs_inner = inner_product_logits.abs().float()
    abs_residual = residual_adjustment.abs().float()
    inner_median = torch.quantile(abs_inner, 0.5)
    residual_median = torch.quantile(abs_residual, 0.5)
    median_ratio = residual_median / inner_median.clamp_min(1e-12)

    return {
        "inner_abs_median": inner_median.item(),
        "residual_abs_median": residual_median.item(),
        "median_abs_ratio": median_ratio.item(),
        "positive_fraction": (residual_adjustment > 0).float().mean().item(),
        "negative_fraction": (residual_adjustment < 0).float().mean().item()
    }


def get_data_label(data):
    return int(data.y.view(-1)[0].cpu().item())


def compute_class_weights(data_list, num_classes, device):
    labels = [get_data_label(data) for data in data_list]
    if len(labels) == 0:
        return None

    if num_classes is None:
        num_classes = max(labels) + 1

    counts = torch.zeros(num_classes, dtype=torch.float)
    for label in labels:
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} is outside the valid range [0, {num_classes - 1}].")
        counts[label] += 1

    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights.to(device)


def evaluate_model(
    model,
    data_list,
    device,
    criterion_0,
    criterion_1,
    criterion_2,
    split_name='Train',
    classification_weight=1.0
):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    total_reconstruction_loss = 0
    total_classification_loss = 0
    valid_count = 0

    with torch.no_grad():
        for data in data_list:
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                x = data.x.to(device)
                target = data.y.view(-1)
                y_true.append(target.cpu().numpy())
                edge_relation_attr_0 = get_edge_relation_attr(data, mask_0)
                edge_relation_attr_1 = get_edge_relation_attr(data, mask_1)
                edge_symmetric_mask_0 = get_edge_symmetric_mask(data, mask_0)
                edge_symmetric_mask_1 = get_edge_symmetric_mask(data, mask_1)

                recon_adj_0, recon_adj_1, z, y = model(
                    x,
                    train_edge_index_0.to(device),
                    train_edge_index_1.to(device),
                    edge_relation_attr_0.to(device),
                    edge_relation_attr_1.to(device),
                    edge_symmetric_mask_0.to(device),
                    edge_symmetric_mask_1.to(device)
                )

                recon_adj_0 = recon_adj_0.squeeze()
                recon_adj_1 = recon_adj_1.squeeze()
                edge_attr_0 = data.edge_attr[mask_0]
                edge_attr_1 = data.edge_attr[mask_1]

                reconstruction_loss = (
                    0.5 * criterion_0(recon_adj_0, edge_attr_0.to(device)) +
                    0.5 * criterion_1(recon_adj_1, edge_attr_1.to(device))
                )
                classification_loss = criterion_2(y, target.to(device))
                loss = reconstruction_loss + classification_weight * classification_loss
                total_loss += loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_classification_loss += classification_loss.item()
                valid_count += 1
                y_pred.append(y.argmax(dim=1).cpu().numpy())

    if valid_count == 0:
        print(f'{split_name} Loss: inf (no valid samples)')
        return float('inf'), float('inf'), float('inf'), 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / valid_count
    avg_reconstruction_loss = total_reconstruction_loss / valid_count
    avg_classification_loss = total_classification_loss / valid_count
    accuracy = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
    f1_macro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    f1_micro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='micro')
    f1_weighted = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='weighted')

    print(
        f'{split_name} Loss: {avg_loss:.8f} '
        f'(reconstruction: {avg_reconstruction_loss:.8f}, '
        f'classification: {avg_classification_loss:.8f})'
    )
    print(
        f'Accuracy:\t{accuracy:.4f}\t'
        f'F1_score_macro:\t{f1_macro:.4f}\t'
        f'F1_score_micro:\t{f1_micro:.4f}\t'
        f'F1_score_weighted:\t{f1_weighted:.4f}'
    )
    return (
        avg_loss,
        avg_reconstruction_loss,
        avg_classification_loss,
        accuracy,
        f1_macro,
        f1_micro,
        f1_weighted
    )


def trimmed_mean(z_stack, trim_ratio: float = 0.1):
    if z_stack.size(0) == 0:
        raise ValueError("z_stack must contain at least one cell.")
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in the range [0, 0.5).")

    sorted_z, _ = torch.sort(z_stack, dim=0)
    num_cells = z_stack.size(0)
    trim_count = int(num_cells * trim_ratio)
    if trim_count == 0:
        return sorted_z.mean(dim=0)

    trimmed_z = sorted_z[trim_count:num_cells - trim_count]
    return trimmed_z.mean(dim=0)


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
    Knockout operation: remove affected edges and set node features to zero.
    knockout_gene_idx: int or iterable.
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
    Compute each gene embedding's L2 distance before and after knockout for one cell.
    Return None if the knockout gene has no positive edge in the current cell network.
    """
    edge_index_1 = get_positive_edge_index(data.edge_index, data.edge_attr)

    if edge_index_1.shape[1] <= 1:
        # print("Not enough edges")
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

def collect_observed_knockout_scores_single_gene_fast(
    model,
    cell_cache,
    device,
    knockout_gene_idx
):
    """
    Fast observed KO score collection.
    """
    obs_distance_dict = {}

    for item in cell_cache:
        dist = compute_knockout_distances_from_cache(
            model=model,
            cache_item=item,
            knockout_gene_idx=knockout_gene_idx,
            device=device
        )

        if dist is None:
            continue

        label = item["label"]
        if label not in obs_distance_dict:
            obs_distance_dict[label] = []
        obs_distance_dict[label].append(dist)

    return obs_distance_dict

def collect_observed_knockout_scores_single_gene(model, data_list, cell_idx,
                                                 device, knockout_gene_idx, only_correct=True):
    """
    Collect perturbation scores by cell type under a true single-gene knockout.
    The perturbation step only requires a non-empty edge_index_1 and no longer
    requires enough edges in edge_index_0.
    """
    obs_distance_dict = {}

    with torch.no_grad():
        for idx in cell_idx:
            data = data_list[idx]
            edge_index_0, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            edge_index_1, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if edge_index_1.shape[1] <= 1:
                # print("Not enough edges")
                continue

            label = int(data.y.cpu().item())

            if only_correct:
                x = data.x.to(device)
                z = model.encode(x, edge_index_1.to(device))
                y = classify_from_z(model, z, device)
                pred = int(y.argmax(dim=1).cpu().item())
                if pred != label:
                    # print("Incorrect cell-type prediction")
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

def classify_from_z(model, z, device):
    """
    Classify cell type using only z to avoid calling the decoder in model.forward().
    Keep the original pooling + fc1/fc2 logic unchanged.
    """
    z = z.to(device)
    y1, _ = torch.max(z, dim=1)
    y1_1 = y1.unsqueeze(0)
    y2 = F.relu(model.fc1(y1_1))
    y = model.fc2(y2)
    return y


def get_active_genes_from_edge_index(edge_index_1, num_genes=None):
    """
    Return genes that participate in at least one positive edge in the current cell network.
    """
    active = torch.unique(edge_index_1.reshape(-1)).cpu()
    return set(active.tolist())


def knockout_positive_edge_index(edge_index_1, knockout_gene_idx):
    """
    Remove edges involving knockout_gene_idx directly from positive edge_index_1.
    This avoids cloning edge_attr.
    """
    src = edge_index_1[0]
    dst = edge_index_1[1]
    keep_mask = (src != knockout_gene_idx) & (dst != knockout_gene_idx)
    return edge_index_1[:, keep_mask]


def encode_with_positive_edges(model, x, edge_index_1, device):
    with torch.no_grad():
        z = model.encode(x.to(device), edge_index_1.to(device))
    return z.cpu()

def prepare_perturbation_cell_cache(
    model,
    data_list,
    cell_idx,
    device,
    only_correct=True
):
    """
    Precompute information reused during perturbation analysis:
    1. label
    2. x
    3. edge_index_1
    4. original z
    5. active_genes
    """
    model.eval()
    cell_cache = []

    with torch.no_grad():
        for idx in cell_idx:
            data = data_list[idx]

            edge_index_1, _ = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if edge_index_1.shape[1] <= 1:
                continue

            label = int(data.y.cpu().item())

            # Compute the original z only once.
            z = encode_with_positive_edges(
                model=model,
                x=data.x,
                edge_index_1=edge_index_1,
                device=device
            )

            if only_correct:
                y = classify_from_z(model, z, device)
                pred = int(y.argmax(dim=1).cpu().item())
                if pred != label:
                    continue

            active_genes = get_active_genes_from_edge_index(edge_index_1)

            cell_cache.append({
                "idx": idx,
                "label": label,
                "x": data.x.cpu(),
                "edge_index_1": edge_index_1.cpu(),
                "z": z.cpu(),
                "active_genes": active_genes
            })

    print(f"Cached valid perturbation cells: {len(cell_cache)}")
    return cell_cache

def compute_knockout_distances_from_cache(model, cache_item, knockout_gene_idx, device):
    """
    Compute post-KO distances from cached single-cell information.
    Return None if knockout_gene_idx is not in the cell's positive-edge network.
    """
    if knockout_gene_idx not in cache_item["active_genes"]:
        return None

    edge_index_1 = cache_item["edge_index_1"]
    edge_index_1_ko = knockout_positive_edge_index(edge_index_1, knockout_gene_idx)

    if edge_index_1_ko.shape[1] <= 1:
        return None

    x_ko = cache_item["x"].clone()
    x_ko[knockout_gene_idx] = 0.0

    z_ko = encode_with_positive_edges(
        model=model,
        x=x_ko,
        edge_index_1=edge_index_1_ko,
        device=device
    )

    dist = torch.norm(cache_item["z"] - z_ko, p=2, dim=1)
    return dist

def _collect_random_knockout_background_fast_legacy(
    model,
    cell_cache,
    num_genes,
    device,
    num_random_knockouts=500,
    agg='median',
    seed=42,
    candidate_genes=None
):
    """
    Fast random knockout background.

    Uses cell_cache to avoid:
    1. repeated classification
    2. repeated original z computation
    3. repeated edge_attr cloning
    """
    rng = np.random.default_rng(seed)

    if candidate_genes is None:
        # Sample only from genes that appear in positive edges in at least some cells.
        active_union = set()
        for item in cell_cache:
            active_union.update(item["active_genes"])
        candidate_genes = np.array(sorted(active_union), dtype=int)
    else:
        candidate_genes = np.asarray(list(candidate_genes), dtype=int)

    random_bg = {}

    for b in range(num_random_knockouts):
        rand_ko = int(rng.choice(candidate_genes, size=1, replace=False)[0])

        dist_dict_b = {}

        for item in cell_cache:
            dist = compute_knockout_distances_from_cache(
                model=model,
                cache_item=item,
                knockout_gene_idx=rand_ko,
                device=device
            )

            if dist is None:
                continue

            label = item["label"]

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
        random_bg[label] = np.stack(random_bg[label], axis=0)
        # print(f"Label {label}: random background shape = {random_bg[label].shape}")

    return random_bg


def collect_random_knockout_background_fast(
    model,
    cell_cache,
    num_genes,
    device,
    num_random_knockouts=500,
    agg='median',
    seed=42,
    candidate_genes=None
):
    """
    Build one random knockout background per cell type.

    num_random_knockouts is treated as the target number of valid background
    vectors for each cell type, rather than only the number of sampling attempts.
    """
    rng = np.random.default_rng(seed)

    if len(cell_cache) == 0:
        print("Warning: no valid cells available for random knockout background.")
        return {}

    def valid_gene_indices(values):
        valid = set()
        for gene_idx in values:
            try:
                gene_idx = int(gene_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= gene_idx < num_genes:
                valid.add(gene_idx)
        return valid

    global_candidates = None
    if candidate_genes is not None:
        global_candidates = valid_gene_indices(candidate_genes)
        if len(global_candidates) == 0:
            print("Warning: candidate_genes contains no valid gene indices.")
            return {}

    cache_by_label = {}
    for item in cell_cache:
        cache_by_label.setdefault(item["label"], []).append(item)

    random_bg = {}

    for label, label_items in cache_by_label.items():
        active_union = set()
        for item in label_items:
            active_union.update(valid_gene_indices(item["active_genes"]))

        if global_candidates is None:
            label_candidates = np.array(sorted(active_union), dtype=int)
        else:
            label_candidates = np.array(sorted(active_union & global_candidates), dtype=int)

        if label_candidates.size == 0:
            print(f"Warning: label {label} has no candidate genes for random knockout background.")
            continue

        label_bg = []
        attempts = 0
        max_attempts = num_random_knockouts * 10

        while len(label_bg) < num_random_knockouts and attempts < max_attempts:
            attempts += 1
            rand_ko = int(rng.choice(label_candidates))

            dist_list = []
            for item in label_items:
                dist = compute_knockout_distances_from_cache(
                    model=model,
                    cache_item=item,
                    knockout_gene_idx=rand_ko,
                    device=device
                )
                if dist is not None:
                    dist_list.append(dist)

            if len(dist_list) == 0:
                continue

            agg_b = aggregate_distances_by_label({label: dist_list}, agg=agg)
            if label in agg_b:
                label_bg.append(agg_b[label])

            print("\r", end="")
            print(
                f"Random knockout background label {label}: "
                f"{len(label_bg)}/{num_random_knockouts}",
                end=""
            )

        print()

        if len(label_bg) == 0:
            print(f"Warning: label {label} collected no valid random knockout backgrounds.")
            continue

        if len(label_bg) < num_random_knockouts:
            print(
                f"Warning: label {label} collected only "
                f"{len(label_bg)}/{num_random_knockouts} random knockout backgrounds "
                f"after {attempts} attempts."
            )

        random_bg[label] = np.stack(label_bg, axis=0)

    return random_bg


def get_significant_gene_idx_from_background(obs_scores, random_bg, cut_off=0.05):
    """
    Return only the significant perturbed gene indices for each label.
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

    # Perturbation parameters.
    knockout_gene_idx=None,
    num_random_knockouts: int = 500,
    perturbation_agg: str = 'mean',
    perturbation_fdr_alpha: float = 0.05,
    only_correct_cells_for_perturbation: bool = True,
    patience: int = 5,
    min_delta: float = 1e-4,
    classification_weight: float = 1.0,
    classification_accuracy: float = 0.0
):
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if classification_weight < 0:
        raise ValueError("classification_weight must be non-negative.")
    if not 0 <= classification_accuracy <= 1:
        raise ValueError("classification_accuracy must be between 0 and 1.")

    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation_dim = get_edge_relation_attr(data_list[0]).shape[1]

    for data in data_list[1:]:
        if get_edge_relation_attr(data).shape[1] != relation_dim:
            raise ValueError("All cell graphs must use the same relation-type vocabulary.")

    model = GraphAutoencoder(
        in_channels1=data_list[0].num_node_features,
        in_channels2=data_list[0].num_nodes,
        hidden_channels=embedding_dim,
        out_channels=out_dim,
        num_relations=relation_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_0 = torch.nn.BCEWithLogitsLoss()
    criterion_1 = torch.nn.BCEWithLogitsLoss()
    class_weights = compute_class_weights(
        data_list=data_list,
        num_classes=out_dim,
        device=device
    )
    criterion_2 = torch.nn.CrossEntropyLoss(weight=class_weights)
    # print(f"Classification loss weight: {classification_weight:.4f}")

    min_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    count_no_improve = 0

    # =========================
    # Keep the original training flow.
    # Training still applies the edge_index_0 / edge_index_1 dual filtering.
    # =========================
    for epoch in range(num_epochs):
        model.train()
        print('epoch: %d' % epoch)

        epoch_indices = list(range(len(data_list)))
        random.Random(seed + epoch).shuffle(epoch_indices)

        for data_index in epoch_indices:
            data = data_list[data_index]
            train_edge_index_0, mask_0 = filter_edges_by_weight(data.edge_index, data.edge_attr, 0)
            train_edge_index_1, mask_1 = filter_edges_by_weight(data.edge_index, data.edge_attr, 1)

            if train_edge_index_0.shape[1] > 1 and train_edge_index_1.shape[1] > 1:
                optimizer.zero_grad()

                x = data.x.to(device)
                edge_relation_attr_0 = get_edge_relation_attr(data, mask_0)
                edge_relation_attr_1 = get_edge_relation_attr(data, mask_1)
                edge_symmetric_mask_0 = get_edge_symmetric_mask(data, mask_0)
                edge_symmetric_mask_1 = get_edge_symmetric_mask(data, mask_1)
                recon_adj_0, recon_adj_1, z, y = model(
                    x,
                    train_edge_index_0.to(device),
                    train_edge_index_1.to(device),
                    edge_relation_attr_0.to(device),
                    edge_relation_attr_1.to(device),
                    edge_symmetric_mask_0.to(device),
                    edge_symmetric_mask_1.to(device)
                )

                recon_adj_0 = recon_adj_0.squeeze()
                recon_adj_1 = recon_adj_1.squeeze()
                edge_attr_0 = data.edge_attr[mask_0]
                edge_attr_1 = data.edge_attr[mask_1]

                target = data.y.view(-1)
                loss_0 = criterion_0(recon_adj_0, edge_attr_0.to(device))
                loss_1 = criterion_1(recon_adj_1, edge_attr_1.to(device))
                loss_2 = criterion_2(y, target.to(device))
                loss = 0.5 * loss_0 + 0.5 * loss_1 + classification_weight * loss_2

                loss.backward()
                optimizer.step()

        (
            total_loss,
            reconstruction_loss,
            classification_loss,
            accuracy,
            f1_macro,
            f1_micro,
            f1_weighted
        ) = evaluate_model(
            model,
            data_list,
            device,
            criterion_0,
            criterion_1,
            criterion_2,
            split_name='Train',
            classification_weight=classification_weight
        )

        if total_loss < min_loss - min_delta:
            count_no_improve = 0
            min_loss = total_loss
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            count_no_improve += 1

        if count_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no full-data total loss improvement > {min_delta} "
                f"for {patience} epochs)"
            )
            break

    if best_model_state is not None:
        model.load_state_dict({
            key: value.to(device)
            for key, value in best_model_state.items()
        })
        print(
            f"Restored best model from epoch {best_epoch + 1} "
            f"with full-data total loss {min_loss:.8f}"
        )
        print(
            "Learned relation-aware residual scale: "
            f"{model.decoder.residual_scale.detach().cpu().item():.6f}"
        )
        print(
            "Maximum absolute residual adjustment: "
            f"{model.decoder.max_residual_adjustment:.6f}"
        )

    # =========================
    # Original evaluation: reconstruct cell-type-specific networks.
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
                edge_relation_attr_0 = get_edge_relation_attr(data, mask_0)
                edge_relation_attr_1 = get_edge_relation_attr(data, mask_1)
                edge_symmetric_mask_0 = get_edge_symmetric_mask(data, mask_0)
                edge_symmetric_mask_1 = get_edge_symmetric_mask(data, mask_1)
                recon_adj_0, recon_adj_1, z, y = model(
                    x,
                    train_edge_index_0.to(device),
                    train_edge_index_1.to(device),
                    edge_relation_attr_0.to(device),
                    edge_relation_attr_1.to(device),
                    edge_symmetric_mask_0.to(device),
                    edge_symmetric_mask_1.to(device)
                )

                label_pred = int(y.argmax(dim=1).cpu().item())
                y_pred.append(label_pred)

                latent_adj = z.cpu()
                if label == label_pred:
                    if label not in z_dict:
                        z_dict[label] = []
                    z_dict[label].append(latent_adj)

    edge_index = data_list[0].edge_index
    edge_relation_attr = get_edge_relation_attr(data_list[0])
    edge_symmetric_mask = get_edge_symmetric_mask(data_list[0])
    adj_dict_mean = {}
    for label, z_list in z_dict.items():
        print(f"Correctly classified cells for label {label}: {len(z_list)}")
        if len(z_list) >= min_cell_count:
            z_stack = torch.stack(z_list)
            joint_z_mean = trimmed_mean(z_stack)

            with torch.no_grad():
                (
                    recon_logits_mean,
                    inner_product_logits,
                    residual_adjustment
                ) = model.decoder.compute_logit_components(
                    joint_z_mean.to(device),
                    edge_index.to(device),
                    edge_relation_attr.to(device),
                    edge_symmetric_mask.to(device)
                )
                recon_adj_mean = torch.sigmoid(recon_logits_mean).cpu().numpy()
                residual_summary = summarize_decoder_adjustment(
                    inner_product_logits,
                    residual_adjustment
                )

            if residual_summary is not None:
                label_name = (
                    cell_labels.get(label, label)
                    if hasattr(cell_labels, 'get')
                    else label
                )
                print(
                    f"Decoder residual diagnostics for {label_name}: "
                    f"median|inner|={residual_summary['inner_abs_median']:.6f}, "
                    f"median|residual|={residual_summary['residual_abs_median']:.6f}, "
                    f"median ratio={residual_summary['median_abs_ratio']:.6f}, "
                    f"positive={residual_summary['positive_fraction']:.4f}, "
                    f"negative={residual_summary['negative_fraction']:.4f}"
                )

            row, col = edge_index.cpu().numpy()
            adj_matrix_mean = coo_matrix(
                (recon_adj_mean, (row, col)),
                shape=(data_list[0].x.shape[0], data_list[0].x.shape[0])
            )
            adj_dict_mean[label] = adj_matrix_mean

    if len(y_true) == 0:
        print("Warning: no valid cells were available for final reconstruction evaluation.")
        accuracy, f1_macro, f1_micro, f1_weighted = 0.0, 0.0, 0.0, 0.0
    else:
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy:\t{accuracy:.4f}\tF1_score_macro:\t{f1_macro:.4f}\t'
          f'F1_score_micro:\t{f1_micro:.4f}\tF1_score_weighted:\t{f1_weighted:.4f}')

    if accuracy < classification_accuracy:
        print(
            f"Skip perturbation analysis because accuracy {accuracy:.4f} "
            f"is below threshold {classification_accuracy:.4f}."
        )
        return accuracy, {}, {}

    # =========================
    # single-cell resolved perturbation analysis
    # 1. Build one shared random knockout background.
    # 2. Then loop over each true knockout.
    # 3. perturbation_results[label][ko_gene][target_gene] = perturbation score
    # =========================
    perturbation_results = {}

    if knockout_gene_idx is not None and len(knockout_gene_idx) > 0:
        print('Start building random knockout background...')
        cell_cache = prepare_perturbation_cell_cache(
            model=model,
            data_list=data_list,
            cell_idx=cell_idx,
            device=device,
            only_correct=only_correct_cells_for_perturbation
        )

        shared_random_bg = collect_random_knockout_background_fast(
            model=model,
            cell_cache=cell_cache,
            num_genes=data_list[0].x.shape[0],
            device=device,
            num_random_knockouts=num_random_knockouts,
            agg=perturbation_agg,
            seed=seed
        )

        for ko_gene in knockout_gene_idx:
            print("\r", end="")
            print(f"Knockout analysis for receptor: {gene_names[ko_gene]}", end="")

            obs_distance_dict = collect_observed_knockout_scores_single_gene_fast(
                model=model,
                cell_cache=cell_cache,
                device=device,
                knockout_gene_idx=ko_gene
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
                label_scores = obs_scores[label]
                perturbation_results[cell_type][gene_names[ko_gene]] = {
                    gene_names[i]: float(label_scores[i])
                    for i in significant_gene_idx
                }
        print()

    return accuracy, adj_dict_mean, perturbation_results
