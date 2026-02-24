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
    n1, n2 = data.shape  # n1是基因数，n2是细胞数
    for k in range(n2):
        feature = data[:, k].reshape(n1, 1)
        string_series = pd.Series(gene_list, name='Gene')
        concatenated_df = pd.DataFrame(feature, columns=[f'Feature_{i}' for i in range(feature.shape[1])])
        final_df = pd.concat([string_series, concatenated_df], axis=1)
        final_df.to_csv(f"./result/feature_file/{cell_list[k]}.txt", sep='\t', index=False)


def csnet(adata, activated_genes, path_matrix):

    # output_feature(adata)
    data = adata.X.T.toarray() if sp.issparse(adata.X) else adata.X.T

    cell_ids = adata.obs['cellid']
    n1, n2 = data.shape  # n1是基因数，n2是细胞数

    result: list[coo_matrix] = []
    sum_net = coo_matrix((n1, n1), dtype=int)
    for k in range(n2):
        gene_set = activated_genes[k]
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

    # 全局 edge_index
    edge_index = torch.tensor(np.vstack((sum_rows, sum_cols)), dtype=torch.long)

    # 构造一个 map: key -> 全局边在 sum_net 中的位置
    # key 采用 tuple (row, col)
    global_edge_map = {(r, c): idx for idx, (r, c) in enumerate(zip(sum_rows, sum_cols))}

    data_list = []

    for k in range(n2):
        feature = torch.tensor(data[:, k].reshape(n1, 1), dtype=torch.float)

        net = result[k]
        # 当前 net 的所有边 (稀疏)
        net_edges = zip(net.row, net.col)

        # 初始化 edge_attr 全 0
        # 注意我们用 float32 更省内存
        edge_attr = torch.zeros(len(sum_rows), dtype=torch.float)

        # 对每个细胞 net 边，在 global map 中打标为 1
        for (r, c) in net_edges:
            if (r, c) in global_edge_map:
                edge_attr[global_edge_map[(r, c)]] = 1.0

        label = torch.tensor(cell_ids.iloc[k], dtype=torch.long)

        torch_data = Data(x=feature, edge_index=edge_index, edge_attr=edge_attr, y=label)
        data_list.append(torch_data)
        del edge_attr
        del feature
        del torch_data

        print(f"\rCell-specific network pruning: {k + 1} / {n2}", end="")
    print()

    return data_list