from decimal import Decimal
import math
import os
import sys
import torch
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from torch_geometric.data import Data


SYMMETRIC_RELATION_TYPES = frozenset({"interacts-with", "in-complex-with"})


def build_edge_relation_attributes(rows, cols, pathway_type):
    """Encode all curated relation labels for each directed edge as multi-hot features."""
    if pathway_type is None:
        return (
            torch.ones((len(rows), 1), dtype=torch.float),
            ("interaction",),
            torch.zeros(len(rows), dtype=torch.bool)
        )

    edge_relations = []
    relation_names = set()
    has_unspecified = False

    for row, col in zip(rows, cols):
        relations = pathway_type[int(row), int(col)]
        if relations is None:
            normalized = ()
        elif isinstance(relations, (set, list, tuple)):
            normalized = tuple(sorted(str(relation) for relation in relations))
        else:
            normalized = (str(relations),)

        if len(normalized) == 0:
            has_unspecified = True
        else:
            relation_names.update(normalized)
        edge_relations.append(normalized)

    if has_unspecified or len(relation_names) == 0:
        relation_names.add("unspecified")

    relation_names = tuple(sorted(relation_names))
    relation_to_index = {
        relation: index
        for index, relation in enumerate(relation_names)
    }
    edge_relation_attr = torch.zeros(
        (len(edge_relations), len(relation_names)),
        dtype=torch.float
    )
    edge_symmetric_mask = torch.zeros(len(edge_relations), dtype=torch.bool)

    for edge_index, relations in enumerate(edge_relations):
        if len(relations) == 0:
            relations = ("unspecified",)
        for relation in relations:
            edge_relation_attr[edge_index, relation_to_index[relation]] = 1.0
        edge_symmetric_mask[edge_index] = set(relations).issubset(SYMMETRIC_RELATION_TYPES)

    return edge_relation_attr, relation_names, edge_symmetric_mask


def output_graph(array, cell_id):
    model_dir = './result/initial_network'
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f'{cell_id}.txt'), 'w') as file:
        file.write(f'{array.shape[0]} {array.shape[1]}\n')
        for r, c, d in zip(array.row, array.col, array.data):
            file.write(f'{r} {c} {d}\n')


def coo_transform(array):
    rows, cols = array.nonzero()
    nonzero_data = [1] * len(rows)
    transformed = coo_matrix((nonzero_data, (rows, cols)), shape=array.shape)
    return transformed


def output_feature(adata):
    os.makedirs('./result/feature_file', exist_ok=True)
    gene_list = adata.var_names
    cell_list = adata.obs_names
    data = adata.X.T.toarray()
    n1, n2 = data.shape  # n1 is the number of genes; n2 is the number of cells.
    for k in range(n2):
        feature = data[:, k].reshape(n1, 1)
        string_series = pd.Series(gene_list, name='Gene')
        concatenated_df = pd.DataFrame(feature, columns=[f'Feature_{i}' for i in range(feature.shape[1])])
        final_df = pd.concat([string_series, concatenated_df], axis=1)
        final_df.to_csv(f"./result/feature_file/{cell_list[k]}.txt", sep='\t', index=False)


def csnet(adata, activated_genes, path_matrix, pathway_type=None):

    # output_feature(adata)
    data = adata.X.T.toarray() if sp.issparse(adata.X) else adata.X.T

    cell_ids = adata.obs['cellid']
    n1, n2 = data.shape  # n1 is the number of genes; n2 is the number of cells.

    result: list[coo_matrix] = []
    sum_net = coo_matrix((n1, n1), dtype=int)
    for k in range(n2):
        gene_set = np.asarray(list(activated_genes[k]), dtype=int)
        csn = np.zeros((n1, n1))
        for i in gene_set:
            for j in gene_set:
                if j != i and data[i, k] > 0 and data[j, k] > 0:
                    csn[i, j] = 1
        csn = csr_matrix(csn)

        print("\r", end="")
        print(f"Cell-specific network inference: {k+1} / {n2}", end="")
        sys.stdout.flush()
        # time.sleep(0.1)

        # direct interactions
        union = csn.multiply(path_matrix).tocoo()

        # mask = union.col > union.row
        # transformed = coo_matrix((union.data[mask], (union.row[mask], union.col[mask])), shape=union.shape)
        transformed = coo_transform(union)
        result.append(transformed)
        sum_net += transformed
    print()

    sum_net = coo_transform(sum_net)

    sum_rows = sum_net.row
    sum_cols = sum_net.col

    # Global edge_index.
    edge_index = torch.tensor(np.vstack((sum_rows, sum_cols)), dtype=torch.long)
    edge_relation_attr, relation_names, edge_symmetric_mask = build_edge_relation_attributes(
        sum_rows,
        sum_cols,
        pathway_type
    )

    # Map each edge key to its global position in sum_net.
    # The key is a tuple: (row, col).
    global_edge_map = {(r, c): idx for idx, (r, c) in enumerate(zip(sum_rows, sum_cols))}

    data_list = []

    for k in range(n2):
        feature = torch.tensor(data[:, k].reshape(n1, 1), dtype=torch.float)

        net = result[k]
        # All edges in the current sparse net.
        net_edges = zip(net.row, net.col)

        # Initialize edge_attr with zeros.
        # Use float32 to reduce memory usage.
        edge_attr = torch.zeros(len(sum_rows), dtype=torch.float)

        # Mark each current cell-net edge in the global map.
        for (r, c) in net_edges:
            if (r, c) in global_edge_map:
                edge_attr[global_edge_map[(r, c)]] = 1.0

        label = torch.tensor(cell_ids.iloc[k], dtype=torch.long)

        torch_data = Data(
            x=feature,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_relation_attr=edge_relation_attr,
            edge_symmetric_mask=edge_symmetric_mask,
            y=label
        )
        torch_data.relation_names = relation_names
        data_list.append(torch_data)
        del edge_attr
        del feature
        del torch_data

        print(f"\rCell-specific network pruning: {k + 1} / {n2}", end="")
    print()

    return data_list
