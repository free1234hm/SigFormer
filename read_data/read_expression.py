import csv
import numpy as np
from scipy.sparse import csr_matrix


def read_sparse_matrix(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)
        line1 = next(file)
        header = line1.strip().split(' ')
        n_rows = int(header[0])
        n_cols = int(header[1])

        data = []
        row_indices = []
        col_indices = []

        for line in file:
            row, col, value = line.strip().split(' ')
            try:
                value = float(value)  # Attempt to convert the value to float
                row_indices.append(int(row) - 1)  # Convert to 0-based indexing
                col_indices.append(int(col) - 1)  # Convert to 0-based indexing
                data.append(value)
            except ValueError:
                print(f"Skipping invalid value: {value} in row: {row}, col: {col}")

        sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))
        sparse_matrix = sparse_matrix.T
        return sparse_matrix


def read_dense_matrix(file_path):
    sample_names = []  # One-dimensional vector for sample names
    gene_names = []  # One-dimensional vector for gene names
    expression_matrix = []  # Two-dimensional array for expression values

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')  # Assuming tab-separated values

        # Read the header row (gene names)
        sample_names = next(reader)[1:]
        # Remove quotes from all elements in sample_names
        sample_names = [name.replace('"', '').replace("'", '') for name in sample_names]

        for row in reader:
            gene_names.append(row[0].replace('"', '').replace("'", ''))  # Extract sample name
            expression_matrix.append([convert2float(value) for value in row[1:]])
    data = np.array(expression_matrix).T

    return sample_names, gene_names, data


def convert2float(value):
    try:
        # Attempt to convert the value to a float
        float_value = float(value)
        return float_value
    except ValueError:
        # If conversion fails, output 0
        return 0