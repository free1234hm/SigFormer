import os
import numpy as np
from scipy.sparse import csr_matrix


def pathway(directory, gene_dict, col1_index: int = 0, col2_index: int = 1, col3_index: int = 2):
    n1 = len(gene_dict)
    pair = np.zeros((n1, n1), dtype=int)
    type = np.empty((n1, n1), dtype=object)
    if os.path.isfile(directory):  # Check if it's a file (not a directory)
        with open(directory, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                elements = line.strip().split('\t')  # Split each line into elements based on delimiter \t
                gene1 = elements[col1_index]
                gene2 = elements[col2_index]
                c3 = elements[col3_index]
                if gene1 in gene_dict and gene2 in gene_dict:
                    idx1, idx2 = gene_dict[gene1], gene_dict[gene2]
                    pair[idx1, idx2] = 1
                    pair[idx2, idx1] = 1
                    if type[idx1, idx2] is None:
                        type[idx1, idx2] = set()
                        type[idx1, idx2].add(c3)
                    else:
                        type[idx1, idx2].add(c3)

                    if c3 == 'interacts-with' or c3 == 'in-complex-with':
                        if type[idx2, idx1] is None:
                            type[idx2, idx1] = set()
                            type[idx2, idx1].add(c3)
                        else:
                            type[idx2, idx1].add(c3)

        np.fill_diagonal(pair, 0)  # setting all the elements on the main diagonal to zero

        return csr_matrix(pair), type
    else:
        print("Error: The reference network file does not exist.")
        return csr_matrix(pair), type


def pathway2(directory, gene_dict, col1_index: int = 0, col2_index: int = 1, col3_index: int = 2):
    n1 = len(gene_dict)
    pair = np.zeros((n1, n1), dtype=int)
    type = np.empty((n1, n1), dtype=object)
    tf_dict = {}
    if os.path.isfile(directory):  # Check if it's a file (not a directory)
        with open(directory, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                elements = line.strip().split('\t')  # Split each line into elements based on delimiter \t
                gene1 = elements[col1_index]
                gene2 = elements[col2_index]
                c3 = elements[col3_index]
                if gene1 in gene_dict and gene2 in gene_dict:
                    idx1, idx2 = gene_dict[gene1], gene_dict[gene2]
                    pair[idx1, idx2] = 1
                    if type[idx1, idx2] is None:
                        type[idx1, idx2] = set()
                        type[idx1, idx2].add(c3)
                    else:
                        type[idx1, idx2].add(c3)

                    if c3 == 'controls-expression-of':
                        if idx1 in tf_dict:
                            tgs = tf_dict[idx1]
                        else:
                            tgs = set()
                        tgs.add(idx2)
                        tf_dict[idx1] = tgs

                    if c3 == 'interacts-with' or c3 == 'in-complex-with':
                        pair[idx2, idx1] = 1
                        if type[idx2, idx1] is None:
                            type[idx2, idx1] = set()
                            type[idx2, idx1].add(c3)
                        else:
                            type[idx2, idx1].add(c3)
        np.fill_diagonal(pair, 0)  # setting all the elements on the main diagonal to zero

        return csr_matrix(pair), type, tf_dict
    else:
        print("Error: The reference network file does not exist.")
        return csr_matrix(pair), type, tf_dict


def ligand_receptor(directory, gene_dict, col1_index: int = 0, col2_index: int = 1, col3_index: int = 2, min_score: int = 1):
    lg_rp = {}
    if os.path.isfile(directory):  # Check if it's a file (not a directory)
        with open(directory, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                elements = line.strip().split('\t')  # Split each line into elements based on delimiter \t
                gene1 = elements[col1_index]
                gene2 = elements[col2_index]
                score = elements[col3_index]
                if int(score) >= min_score and gene1 in gene_dict and gene2 in gene_dict:
                    idx1, idx2 = gene_dict[gene1], gene_dict[gene2]
                    if idx1 not in lg_rp:
                        lg_rp[idx1] = set()  # Create a new set if it doesn't exist
                    lg_rp[idx1].add(idx2)
        return lg_rp
    else:
        print("Error: The reference network file does not exist.")
        return lg_rp