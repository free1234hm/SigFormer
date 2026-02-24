import numpy as np
from scipy.sparse import coo_matrix


def create_unified_gene_list(sum_gene_lists):
    # Create a unified gene list (set of unique genes)
    unified_genes = set()
    for cell, gene_list in sum_gene_lists.items():
        unified_genes.update(gene_list)
    return sorted(unified_genes)


def gene_to_index_mapping(unified_gene_list, gene_list):
    # Create a mapping from the original gene list to the unified gene list
    return {gene: i for i, gene in enumerate(unified_gene_list)}


def reindex_adjacency_matrix(adj_matrix, original_gene_list, unified_gene_list):
    # Create a mapping from original genes to unified gene list
    gene_mapping = gene_to_index_mapping(unified_gene_list, original_gene_list)

    # Reindex the rows and columns of the adjacency matrix according to the unified gene list
    row_indices = np.array([gene_mapping[gene] for gene in original_gene_list])
    col_indices = np.array([gene_mapping[gene] for gene in original_gene_list])

    # The adj_matrix is a coo_matrix, so we need to map its row/col indices
    new_row_indices = row_indices[adj_matrix.row]
    new_col_indices = col_indices[adj_matrix.col]

    # Return a new coo_matrix with the reindexed rows and columns
    return coo_matrix((adj_matrix.data, (new_row_indices, new_col_indices)),
                      shape=(len(unified_gene_list), len(unified_gene_list)))


def align_adjacency_matrices(sum_matrix, sum_gene_lists):
    # Create a unified gene list from all the gene lists
    unified_gene_list = create_unified_gene_list(sum_gene_lists)

    # Reindex each adjacency matrix to match the unified gene list
    final_matrix = {}
    for cell, adj_matrix in sum_matrix.items():
        gene_list = sum_gene_lists[cell]
        reindexed_adj_matrix = reindex_adjacency_matrix(adj_matrix, gene_list, unified_gene_list)
        final_matrix[cell] = reindexed_adj_matrix

    return final_matrix, unified_gene_list
