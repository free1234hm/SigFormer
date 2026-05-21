import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix


def integrate_multiple_graphs(cell, adj_matrices, gene_lists, num_data):
    unified_genes = sorted(set(gene for genelist in gene_lists for gene in genelist))
    gene_to_index = {gene: idx for idx, gene in enumerate(unified_genes)}
    n_genes = len(unified_genes)

    edge_counts = {}
    edge_weights = {}  # 新增：用于记录权重的累加和

    for adj, genelist in zip(adj_matrices, gene_lists):
        rows, cols, values = adj.row, adj.col, adj.data  # 新增：提取 values
        for row, col, val in zip(rows, cols, values):
            source_gene = gene_to_index[genelist[row]]
            target_gene = gene_to_index[genelist[col]]
            edge = (source_gene, target_gene)
            if edge in edge_counts:
                edge_counts[edge] += 1
                edge_weights[edge] += val  # 累加权重
            else:
                edge_counts[edge] = 1
                edge_weights[edge] = val  # 初始化权重

    rows, cols, data = [], [], []

    for (r, c), v in edge_counts.items():
        if num_data == 1 and v > 0:
            rows.append(r)
            cols.append(c)
            data.append(edge_weights[(r, c)] / v)  # 修改：保存平均权重，而不是出现次数 v
        elif num_data > 1 and v >= min(10, max(2, num_data / 2)):
            rows.append(r)
            cols.append(c)
            data.append(edge_weights[(r, c)] / v)  # 修改：保存平均权重

    integrated_adj = coo_matrix((data, (rows, cols)), shape=(n_genes, n_genes))
    return integrated_adj, unified_genes


def integrate_multiple_dicts(all_perturbation_results):
    for cell_type in all_perturbation_results:
        for ko_gene in all_perturbation_results[cell_type]:
            gene_lists = all_perturbation_results[cell_type][ko_gene]
            if len(gene_lists) == 1:
                all_perturbation_results[cell_type][ko_gene] = gene_lists[0]
            else:
                flat_genes = [g for lst in gene_lists for g in (lst if isinstance(lst, (list, tuple)) else [lst])]
                counts = Counter(flat_genes)
                merged_genes = [g for g, c in counts.items() if c >= min(10, max(2, len(gene_lists)/2))]
                all_perturbation_results[cell_type][ko_gene] = merged_genes

    return all_perturbation_results


def get_consensus_graphs(adj_matrices, gene_lists, threshold):
    # Step 1: Create a unified gene set and mapping
    unified_genes = sorted(set(gene for genelist in gene_lists for gene in genelist))
    gene_to_index = {gene: idx for idx, gene in enumerate(unified_genes)}
    n_genes = len(unified_genes)

    # Step 2: Extract edges from all graphs
    edge_counts = {}
    for adj, genelist in zip(adj_matrices, gene_lists):
        rows, cols = adj.row, adj.col
        for row, col in zip(rows, cols):
            source_gene = genelist[row]
            target_gene = genelist[col]
            edge = (source_gene, target_gene)
            if edge in edge_counts:
                edge_counts[edge] += 1
            else:
                edge_counts[edge] = 1

    # Step 3: Filter edges that appear more than twice
    filtered_edges = [(gene_to_index[edge[0]], gene_to_index[edge[1]])
                      for edge, count in edge_counts.items() if count >= threshold]

    # Step 4: Build the integrated adjacency matrix
    if filtered_edges:
        rows, cols = zip(*filtered_edges)
        integrated_adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_genes, n_genes))
    else:
        # Handle case where no edges meet the criteria
        integrated_adj = coo_matrix((n_genes, n_genes))

    return integrated_adj, unified_genes

# Example usage:
# adj_matrices = [adj_G1, adj_G2, adj_G3, ...]  # List of coo_matrices
# gene_lists = [genelist1, genelist2, genelist3, ...]  # Corresponding gene lists
# integrated_adj, unified_genes = integrate_multiple_graphs(adj_matrices, gene_lists)
