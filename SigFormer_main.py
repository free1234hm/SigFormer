import os
import sys
import argparse
import csv
import gc
import anndata
import time
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from scipy.sparse import coo_matrix
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from csn.GAE_all_train import graph_processing
from csn.cell_net import csnet
from read_data.read_interaction import pathway2, ligand_receptor
from csn.Integrate_graphs import integrate_multiple_graphs, integrate_multiple_dicts
from csn.Align_matrices import align_adjacency_matrices
from csn.Infer_pathway import infer_pathway


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false, yes/no, or 1/0")


def read_background_gene_set(file_path):
    """Read background genes from the first column of a text, TSV, or CSV file."""
    header_names = {
        "gene", "genes", "gene_name", "gene_names", "gene_symbol", "gene_symbols",
        "feature", "features",
    }
    genes = []
    seen = set()

    with open(file_path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in line:
                gene = line.split("\t", 1)[0].strip()
            elif "," in line:
                gene = next(csv.reader([line]))[0].strip()
            else:
                gene = line.split()[0].strip()

            if not gene:
                continue
            if not genes and gene.casefold() in header_names:
                continue
            if gene not in seen:
                genes.append(gene)
                seen.add(gene)

    if not genes:
        raise ValueError(
            f"No background genes were found in the first column of {file_path}"
        )

    return genes


def read_retained_cell_types(file_path):
    """Read exact cell-type labels, preserving spaces within each label."""
    cell_types = []
    seen = set()
    with open(file_path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                cell_type = next(csv.reader([line], delimiter="\t"))[0].strip()
            elif "," in line:
                cell_type = next(csv.reader([line]))[0].strip()
            else:
                cell_type = line
            if not cell_type:
                continue
            if not cell_types and cell_type.casefold() in {"celltype", "cell_type"}:
                continue
            if cell_type not in seen:
                cell_types.append(cell_type)
                seen.add(cell_type)
    if not cell_types:
        raise ValueError(f"No cell types were found in {file_path}")
    return cell_types


def safe_filename_component(value):
    invalid_filename_chars = set('<>:"/\\|?*')
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    safe_value = "".join(
        "_" if char in invalid_filename_chars else char
        for char in str(value)
    ).strip().rstrip(".")
    if not safe_value:
        safe_value = "unnamed_celltype"
    if safe_value.upper() in reserved_names:
        safe_value = f"_{safe_value}"
    return safe_value


def save_cell_specific_networks(final_matrix, gene_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    used_filenames = set()

    for cell, adjacency in final_matrix.items():
        safe_cell = safe_filename_component(cell)

        base_name = safe_cell
        suffix = 2
        while safe_cell.casefold() in used_filenames:
            safe_cell = f"{base_name}_{suffix}"
            suffix += 1
        used_filenames.add(safe_cell.casefold())

        adjacency = adjacency.tocoo(copy=False)
        output_file = os.path.join(output_dir, f"{safe_cell}.txt")
        with open(output_file, "w", encoding="utf-8", newline="") as handle:
            handle.write("Source\tTarget\tWeight\n")
            for row, col, weight in zip(adjacency.row, adjacency.col, adjacency.data):
                source_gene = gene_list[int(row)]
                target_gene = gene_list[int(col)]
                handle.write(f"{source_gene}\t{target_gene}\t{float(weight)!r}\n")


def read_file(file_path, index_cell, retained_cell_types=None):
    adata = anndata.read_h5ad(file_path)

    if "celltype" not in adata.obs.columns:
        if "cell_type" in adata.obs.columns:
            adata.obs["celltype"] = adata.obs["cell_type"]
            adata.obs.drop(columns="cell_type", inplace=True)
        else:
            print(f"Warning: {file_path.stem}.h5ad must have a .obs['celltype'] attribute")
            sys.exit(1)

    if "sample" not in adata.obs.columns:
        adata.obs["sample"] = "merged"

    adata = adata[adata.obs['celltype'].notna() & (adata.obs['celltype'] != '')].copy()
    if retained_cell_types is not None:
        available_types = set(adata.obs["celltype"])
        missing_types = [ct for ct in retained_cell_types if ct not in available_types]
        if missing_types:
            print(
                f"Warning: requested cell types absent from {file_path.name}: "
                + ", ".join(missing_types)
            )
        initial_count = adata.n_obs
        adata = adata[adata.obs["celltype"].isin(retained_cell_types)].copy()
        print(
            f"Retained {adata.n_obs}/{initial_count} cells across "
            f"{adata.obs['celltype'].nunique()} selected cell types before preprocessing "
            "and background-gene selection."
        )
    if index_cell not in adata.obs['celltype'].values:
        print(f"Warning: adata.obs['celltype'] must contain '{index_cell}'")
        sys.exit(1)

    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    return adata


def _get_nonzero_values(X):
    if sp.issparse(X):
        return X.data
    return np.ravel(X)


def _cell_sums(X):
    return np.asarray(X.sum(axis=1)).ravel()


def detect_preprocessing(adata, norm_tol=0.05, integer_tol=1e-8):
    # Scanpy commonly stores log1p info in adata.uns["log1p"]
    if "log1p" in adata.uns:
        return {
            "normalized": True,
            "log1p": True,
            "certain": True,
            "reason": 'Detected adata.uns["log1p"]'
        }

    # User/tool-defined metadata
    pp_info = adata.uns.get("preprocessing", {})
    if isinstance(pp_info, dict) and ("normalized" in pp_info or "log1p" in pp_info):
        normalized = bool(pp_info.get("normalized", False))
        log1p_done = bool(pp_info.get("log1p", False))
        # Conservative rule: if log1p is explicitly marked, do not normalize again
        if log1p_done:
            normalized = True
        return {
            "normalized": normalized,
            "log1p": log1p_done,
            "certain": True,
            "reason": 'Detected adata.uns["preprocessing"]'
        }

    # ------------------------------------------------------------------
    # 2) Heuristic detection
    # ------------------------------------------------------------------
    X = adata.X
    vals = _get_nonzero_values(X)

    # Empty matrix or all-zero matrix
    if vals.size == 0:
        return {
            "normalized": False,
            "log1p": False,
            "certain": False,
            "reason": "Empty or all-zero matrix"
        }

    min_val = vals.min()
    max_val = vals.max()

    # Whether matrix contains fractional values
    has_fraction = np.any(np.abs(vals - np.round(vals)) > integer_tol)

    # Per-cell sums
    sums = _cell_sums(X)
    finite_sums = sums[np.isfinite(sums) & (sums > 0)]

    if finite_sums.size == 0:
        rel_sd = np.inf
    else:
        rel_sd = np.std(finite_sums) / (np.mean(finite_sums) + 1e-12)

    looks_logged = (
        (min_val >= 0) and
        has_fraction and
        (max_val < 50)
    )

    looks_normalized = rel_sd < norm_tol
    if looks_logged:
        looks_normalized = True

    return {
        "normalized": bool(looks_normalized),
        "log1p": bool(looks_logged),
        "certain": False
    }


def preprocess(
    adata,
    hvg_top_genes,
    background_genes=None,
    min_cell=0.01,
    min_gene=0.01,
    normalize=True,
    log_trans=True,
    target_sum=1e4,
):
    try:
        state = detect_preprocessing(adata)
        # --------------------------------------------------------------
        # Filtering
        # --------------------------------------------------------------
        min_genes = max(1, int(min_gene * adata.n_vars))
        sc.pp.filter_cells(adata, min_genes=min_genes)

        min_cells = max(1, int(min_cell * adata.n_obs))
        sc.pp.filter_genes(adata, min_cells=min_cells)

        # --------------------------------------------------------------
        # Transform only when needed
        # --------------------------------------------------------------
        if normalize:
            if state["log1p"]:
                print("Skip cell normalization: input appears already log-transformed.")
            elif state["normalized"]:
                print("Skip cell normalization: input appears already normalized.")
            else:
                print(f"Run cell normalization (target_sum={target_sum})")
                sc.pp.normalize_total(adata, target_sum=target_sum)

        if log_trans:
            if state["log1p"]:
                print("Skip log transformation: input appears already log-transformed.")
            else:
                print("Run log transformation")
                sc.pp.log1p(adata)

        # --------------------------------------------------------------
        # Background-gene selection
        # A user-provided set takes precedence over data-driven HVG selection.
        # --------------------------------------------------------------
        if background_genes is not None:
            available_genes = set(adata.var_names)
            selected_genes = [gene for gene in background_genes if gene in available_genes]
            missing_count = len(background_genes) - len(selected_genes)

            if not selected_genes:
                raise ValueError(
                    "None of the user-provided background genes remain after gene filtering "
                    "or match adata.var_names."
                )

            return adata[:, selected_genes]

        # For Scanpy's common workflow, flavor="seurat" matches log1p data.
        if adata.n_vars > hvg_top_genes:
            print(f"Select the top {hvg_top_genes} HVGs as background genes.")
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top_genes)
            hvg_genes = adata.var[adata.var['highly_variable']].index
            filter_adata = adata[:, list(hvg_genes)]
            return filter_adata
        else:
            print(
                f"Use all {adata.n_vars} genes as background genes because the dataset "
                f"contains no more than {hvg_top_genes} genes."
            )
            return adata

    except Exception as e:
        print(f"Error in preprocess: {e}")
        raise


def _mode_or_first(values):
    mode = values.mode(dropna=True)
    if len(mode) > 0:
        return mode.iloc[0]
    return values.iloc[0] if len(values) > 0 else None


def _cluster_mean(X, indices, expression_threshold=0.0):
    X_sub = X[indices]
    mean_values = np.asarray(X_sub.mean(axis=0)).ravel()
    if expression_threshold > 0:
        mean_values[mean_values < expression_threshold] = 0.0
    return mean_values


def _compression_embedding(X, n_components=30, random_state=42):
    n_cells, n_features = X.shape
    n_components = min(n_components, n_cells - 1, n_features)

    if n_components < 1:
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    if sp.issparse(X):
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(X)
    else:
        min_shape = min(n_cells, n_features)
        svd_solver = "randomized" if n_components < min_shape else "auto"
        reducer = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
        embedding = reducer.fit_transform(np.asarray(X))

    return embedding.astype(np.float32, copy=False)


def _allocate_metacells_by_target(celltype_groups, target_metacells):
    counts = np.array([len(indices) for _, indices in celltype_groups], dtype=int)
    if counts.size == 0:
        return {}

    min_possible = len(counts)
    max_possible = int(counts.sum())
    target_metacells = int(min(max(target_metacells, min_possible), max_possible))

    exact = counts / counts.sum() * target_metacells
    remainders = exact - np.floor(exact)
    allocations = np.floor(exact).astype(int)
    allocations = np.maximum(allocations, 1)
    allocations = np.minimum(allocations, counts)

    while allocations.sum() < target_metacells:
        candidates = np.where(allocations < counts)[0]
        if len(candidates) == 0:
            break
        order = candidates[np.argsort(-remainders[candidates])]
        for idx in order:
            if allocations.sum() >= target_metacells:
                break
            allocations[idx] += 1

    while allocations.sum() > target_metacells:
        candidates = np.where(allocations > 1)[0]
        if len(candidates) == 0:
            break
        order = candidates[np.argsort(remainders[candidates])]
        for idx in order:
            if allocations.sum() <= target_metacells:
                break
            allocations[idx] -= 1

    return {
        celltype: int(n_metacells)
        for (celltype, _), n_metacells in zip(celltype_groups, allocations)
    }


def compress_anndata(
        adata,
        block_size,
        expression_threshold=0.0,
        clustering_obsm_key=None
    ):
    n_cells = adata.n_obs

    if n_cells <= block_size:
        if "metacell_n_cells" not in adata.obs.columns:
            adata.obs["metacell_n_cells"] = 1
        return adata

    if expression_threshold < 0:
        raise ValueError("expression_threshold must be non-negative.")

    celltype_groups = [
        (celltype, np.asarray(indices, dtype=int))
        for celltype, indices in adata.obs.groupby("celltype", observed=False).indices.items()
        if len(indices) > 0
    ]
    metacell_allocations = _allocate_metacells_by_target(celltype_groups, block_size)
    target_metacells = sum(metacell_allocations.values())
    print(
        f"Compress anndata from {n_cells} cells to ~{target_metacells} metacells "
        f"(target={block_size}) using cell-type-stratified metacells."
    )
    if expression_threshold > 0:
        print(f"Set metacell mean expression values < {expression_threshold} to 0.")

    X = adata.X
    if clustering_obsm_key is None:
        embedding = _compression_embedding(X, n_components=30, random_state=42)
        print("Cluster metacells using the 30-dimensional expression embedding.")
    else:
        if clustering_obsm_key not in adata.obsm:
            raise KeyError(
                f"Cannot find adata.obsm[{clustering_obsm_key!r}] for metacell clustering."
            )
        embedding = np.asarray(adata.obsm[clustering_obsm_key])
        if (
            embedding.ndim != 2
            or embedding.shape[0] != n_cells
            or not np.issubdtype(embedding.dtype, np.number)
        ):
            raise ValueError(
                f"adata.obsm[{clustering_obsm_key!r}] must be a numeric matrix "
                "with one row per cell."
            )
        if not np.isfinite(embedding).all():
            raise ValueError(
                f"adata.obsm[{clustering_obsm_key!r}] contains non-finite values."
            )
        embedding = embedding.astype(np.float32, copy=False)
        print(
            f"Cluster metacells using spatial embedding "
            f"adata.obsm[{clustering_obsm_key!r}]."
        )
    # print(f"Metacell clustering embedding shape: {embedding.shape}")

    obs_records = []
    obs_names = []
    X_rows = []
    obsm_rows = {
        key: []
        for key, value in adata.obsm.items()
        if np.asarray(value).shape[0] == n_cells and np.issubdtype(np.asarray(value).dtype, np.number)
    }

    for celltype, indices in celltype_groups:
        n_type = len(indices)
        n_metacells = metacell_allocations[celltype]

        if n_metacells == n_type:
            labels = np.arange(n_type)
        else:
            batch_size = min(n_type, max(1024, n_metacells * 10))
            kmeans = MiniBatchKMeans(
                n_clusters=n_metacells,
                random_state=42,
                batch_size=batch_size,
                n_init=3,
                max_iter=100
            )
            labels = kmeans.fit_predict(embedding[indices])

        for metacell_idx in range(n_metacells):
            member_positions = np.where(labels == metacell_idx)[0]
            member_indices = indices[member_positions]
            if len(member_indices) == 0:
                continue

            X_rows.append(_cluster_mean(
                X,
                member_indices,
                expression_threshold=expression_threshold
            ).astype(np.float32))

            member_obs = adata.obs.iloc[member_indices]
            record = {
                col: _mode_or_first(member_obs[col])
                for col in adata.obs.columns
            }
            record["celltype"] = celltype
            record["metacell_n_cells"] = int(len(member_indices))
            record["metacell_id"] = f"{celltype}_metacell_{metacell_idx}"
            obs_records.append(record)
            obs_names.append(f"{celltype}_metacell_{metacell_idx}")

            for key in obsm_rows:
                obsm_rows[key].append(np.asarray(adata.obsm[key])[member_indices].mean(axis=0))

    if len(X_rows) == 0:
        print("Warning: metacell compression produced no metacells; using original anndata.")
        adata.obs["metacell_n_cells"] = 1
        return adata

    compressed = anndata.AnnData(
        X=np.vstack(X_rows),
        obs=pd.DataFrame(obs_records, index=obs_names),
        var=adata.var.copy(),
        uns=adata.uns.copy()
    )
    nnz = np.count_nonzero(compressed.X)
    total_values = compressed.X.shape[0] * compressed.X.shape[1]
    print(f"Compressed expression nonzero ratio: {nnz / total_values:.4f}")
    for key, rows in obsm_rows.items():
        if len(rows) == len(obs_records):
            compressed.obsm[key] = np.vstack(rows)

    print(f"Compressed size: {compressed.shape}")
    print("Metacell number of each cell type:")
    print(compressed.obs["celltype"].value_counts())
    return compressed


def split_adjacency_matrix(adj_G, adj_type):
    A_rows, A_cols, A_data = [], [], []
    B_rows, B_cols, B_data = [], [], []

    for i, j, value in zip(adj_G.row, adj_G.col, adj_G.data):
        if value > 0:
            if 'controls-expression-of' in adj_type[i, j]:
                A_rows.append(i)
                A_cols.append(j)
                A_data.append(value)  # Keep the original edge weight.
            if adj_type[i, j] != {'controls-expression-of'}:
                B_rows.append(i)
                B_cols.append(j)
                B_data.append(value)  # Keep the original edge weight.

    A = coo_matrix((A_data, (A_rows, A_cols)), shape=adj_G.shape)
    B = coo_matrix((B_data, (B_rows, B_cols)), shape=adj_G.shape)
    return A, B

# Get the out-degree of all nodes
def save_out_degree(adj_G, gene_list, output_file):
    row_degrees = np.bincount(adj_G.row, minlength=adj_G.shape[0])  # Sum the number of non-zero entries in each row
    with open(output_file, 'w') as f:  # Write to a text file with the specified format
        f.write("TF\tTargets\n")
        for i, degree in enumerate(row_degrees):
            if degree > 0:
                f.write(f"{gene_list[i]}\t{degree}\n")

def save_degree(adj_G, gene_list, output_file):
    out_degrees = np.bincount(adj_G.row, minlength=adj_G.shape[0])  # Calculate out-degree (row-wise counts of non-zero entries)
    in_degrees = np.bincount(adj_G.col, minlength=adj_G.shape[0])  # Calculate in-degree (column-wise counts of non-zero entries)
    with open(output_file, 'w') as f:  # Save the degrees to a text file
        f.write("Gene\tUpstream\tDownstream\tDegree\n")
        for i in range(adj_G.shape[0]):
            if in_degrees[i] > 0 or out_degrees[i] > 0:
                f.write(f"{gene_list[i]}\t{in_degrees[i]}\t{out_degrees[i]}\t{in_degrees[i]+out_degrees[i]}\n")


def _is_numeric_obsm_matrix(value, n_obs):
    arr = np.asarray(value)
    return arr.ndim == 2 and arr.shape[0] == n_obs and np.issubdtype(arr.dtype, np.number)


def get_spatial_obsm_key(adata):
    preferred_keys = ["spatial", "spatial_embeddings"]
    for key in preferred_keys:
        if key in adata.obsm and _is_numeric_obsm_matrix(adata.obsm[key], adata.n_obs):
            return key

    spatial_like_keys = [
        key for key in adata.obsm.keys()
        if "spatial" in str(key).lower() and _is_numeric_obsm_matrix(adata.obsm[key], adata.n_obs)
    ]
    if len(spatial_like_keys) > 0:
        return sorted(spatial_like_keys)[0]

    return None


def find_celltype_knn_indices(
        adata,
        embedding_key="spatial",
        celltype_key="celltype",
        n_neighbors=10,
        metric='euclidean'
    ):
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    labels = np.asarray(adata.obs[celltype_key])
    coordinates = np.asarray(adata.obsm[embedding_key])
    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"adata.obsm[{embedding_key!r}] contains non-finite spatial coordinates."
        )
    cell_knn_indices = {}
    neighbor_indices_by_celltype = {
        celltype: set()
        for celltype in pd.unique(labels)
    }

    if adata.n_obs <= 1:
        return {
            celltype: []
            for celltype in neighbor_indices_by_celltype
        }, cell_knn_indices

    effective_k = min(n_neighbors, adata.n_obs - 1)
    model = NearestNeighbors(
        n_neighbors=min(effective_k + 1, adata.n_obs),
        algorithm="auto",
        metric=metric
    )
    model.fit(coordinates)
    candidate_indices = model.kneighbors(coordinates, return_distance=False)

    for cell_index, candidates in enumerate(candidate_indices):
        neighbors = [
            int(candidate)
            for candidate in candidates
            if int(candidate) != cell_index
        ][:effective_k]
        cell_knn_indices[cell_index] = neighbors

        receiver_celltype = labels[cell_index]
        neighbor_indices_by_celltype[receiver_celltype].update(
            neighbor_index
            for neighbor_index in neighbors
            if labels[neighbor_index] != receiver_celltype
        )

    neighbor_indices_by_celltype = {
        celltype: sorted(indices)
        for celltype, indices in neighbor_indices_by_celltype.items()
    }
    return neighbor_indices_by_celltype, cell_knn_indices


def get_spatial_sender_type_sets(
        adata,
        neighbor_indices_by_celltype,
        cell_knn_indices,
        celltype_key="celltype"
    ):
    labels = np.asarray(adata.obs[celltype_key])
    all_indices = np.arange(adata.n_obs, dtype=int)
    neighbor_sender_types = {}
    distant_sender_types = {}
    distant_indices_by_celltype = {}

    for receiver_celltype, neighbor_indices in neighbor_indices_by_celltype.items():
        nonreceiver_indices = all_indices[labels != receiver_celltype]
        neighbor_indices = np.asarray(neighbor_indices, dtype=int)
        nonneighbor_indices = np.setdiff1d(
            nonreceiver_indices,
            neighbor_indices,
            assume_unique=False
        )
        distant_indices = [
            int(cell_index)
            for cell_index in nonneighbor_indices
            if not any(
                labels[neighbor_index] == receiver_celltype
                for neighbor_index in cell_knn_indices.get(int(cell_index), [])
            )
        ]

        neighbor_sender_types[receiver_celltype] = set(labels[neighbor_indices])
        distant_sender_types[receiver_celltype] = set(labels[distant_indices])
        distant_indices_by_celltype[receiver_celltype] = distant_indices

    return neighbor_sender_types, distant_sender_types, distant_indices_by_celltype


def collect_spatial_ligand_expression_counts(
        adata,
        sender_indices_by_receiver,
        ligand_indices,
        celltype_key="celltype"
    ):
    labels = np.asarray(adata.obs[celltype_key])
    gene_names = np.asarray(adata.var_names)
    ligand_indices = np.asarray(sorted(ligand_indices), dtype=int)
    expression_counts = {}

    if len(ligand_indices) == 0:
        return expression_counts

    for receiver_celltype, candidate_indices in sender_indices_by_receiver.items():
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        if len(candidate_indices) == 0:
            continue

        for sender_celltype in pd.unique(labels[candidate_indices]):
            sender_indices = candidate_indices[
                labels[candidate_indices] == sender_celltype
            ]
            if len(sender_indices) == 0:
                continue

            if sp.issparse(adata.X):
                subset = adata.X[sender_indices][:, ligand_indices]
                positive_counts = np.asarray((subset > 0).sum(axis=0)).ravel()
            else:
                subset = np.asarray(adata.X)[np.ix_(sender_indices, ligand_indices)]
                positive_counts = np.count_nonzero(subset > 0, axis=0)

            pair = (sender_celltype, receiver_celltype)
            expression_counts[pair] = {
                gene_names[gene_index]: [int(positive_count), int(len(sender_indices))]
                for gene_index, positive_count in zip(ligand_indices, positive_counts)
            }

    return expression_counts


def merge_ligand_expression_counts(total_counts, sample_counts):
    for pair, gene_counts in sample_counts.items():
        pair_counts = total_counts.setdefault(pair, {})
        for gene, (positive_count, cell_count) in gene_counts.items():
            if gene not in pair_counts:
                pair_counts[gene] = [0, 0]
            pair_counts[gene][0] += positive_count
            pair_counts[gene][1] += cell_count


def select_spatial_ligands(expression_counts, unified_gene_dict, min_fraction):
    selected_ligands = {}
    for pair, gene_counts in expression_counts.items():
        pair_ligands = {
            unified_gene_dict[gene]
            for gene, (positive_count, cell_count) in gene_counts.items()
            if gene in unified_gene_dict
            and cell_count > 0
            and positive_count / cell_count > min_fraction
        }
        if len(pair_ligands) > 0:
            selected_ligands[pair] = pair_ligands
    return selected_ligands


def normalize_feature_scores(raw_scores):
    """Min-max normalize feature scores to [0, 1] within a cell type."""
    numeric_scores = {
        feature: float(score)
        for feature, score in raw_scores.items()
    }
    if not numeric_scores:
        return {}

    min_score = min(numeric_scores.values())
    max_score = max(numeric_scores.values())
    score_range = max_score - min_score
    if score_range == 0:
        normalized_value = 0.0 if max_score == 0 else 1.0
        return {feature: normalized_value for feature in numeric_scores}

    return {
        feature: (score - min_score) / score_range
        for feature, score in numeric_scores.items()
    }


def normalize_inferred_pathway_ko_scores(inference_batches):
    """Max-scale KO scores over pathways that will actually be written."""
    max_score_by_receiver = {}
    for batch in inference_batches:
        receiver_cell = batch["receiver_cell"]
        for pathway in batch["pathways"]:
            score = float(pathway[-1])
            if np.isfinite(score) and score >= 0:
                max_score_by_receiver[receiver_cell] = max(
                    max_score_by_receiver.get(receiver_cell, 0.0),
                    score
                )

    for batch in inference_batches:
        max_score = max_score_by_receiver.get(batch["receiver_cell"], 0.0)
        for pathway in batch["pathways"]:
            score = float(pathway[-1])
            pathway[-1] = (
                score / max_score
                if max_score > 0 and np.isfinite(score) and score >= 0
                else 0.0
            )

    return inference_batches


def read_celltype_feature_score_file(file_path, feature_label):
    """Read cell type, feature, and optional score columns from a TSV file."""
    raw_scores_by_cell = {}
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [
            (reader.line_num, [element.strip() for element in elements])
            for elements in reader
            if elements
            and any(element.strip() for element in elements)
            and not elements[0].strip().startswith("#")
        ]

    has_score_column = any(len(elements) >= 3 for _, elements in rows)
    first_record = True
    for line_no, elements in rows:
        if len(elements) < 2:
            raise ValueError(
                f"{file_path} line {line_no}: expected at least two "
                f"tab-delimited columns: celltype and {feature_label}"
            )

        cell_type, feature_text = elements[:2]
        score_text = elements[2] if len(elements) >= 3 else ""
        cell_header = cell_type.casefold().replace(" ", "_")
        feature_header = feature_text.casefold().replace(" ", "_")
        if first_record and (
            cell_header in {"celltype", "cell_type"}
            and feature_header in {
                "feature", "gene", "protein", "protein_name", "tf", "tf_name"
            }
        ):
            first_record = False
            continue
        first_record = False

        if not cell_type:
            raise ValueError(
                f"{file_path} line {line_no}: celltype must not be empty"
            )

        features = [
            feature.strip()
            for feature in feature_text.split(";")
            if feature.strip()
        ]
        if not features:
            raise ValueError(
                f"{file_path} line {line_no}: expected at least one "
                f"{feature_label} name in the second column"
            )

        if not has_score_column:
            score = 1.0
        elif score_text == "":
            score = 0.0
        else:
            try:
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(
                    f"{file_path} line {line_no}: invalid {feature_label} score "
                    f"{score_text!r}"
                ) from exc

        if not np.isfinite(score):
            raise ValueError(
                f"{file_path} line {line_no}: {feature_label} score must be finite"
            )

        cell_scores = raw_scores_by_cell.setdefault(cell_type, {})
        for feature in features:
            cell_scores[feature] = max(
                cell_scores.get(feature, float('-inf')),
                score
            )

    if not raw_scores_by_cell:
        raise ValueError(f"No {feature_label} records found in {file_path}")

    return {
        cell_type: normalize_feature_scores(raw_scores)
        for cell_type, raw_scores in raw_scores_by_cell.items()
    }


def read_scproteomics_inputs(scProteomics_path):
    if scProteomics_path is None:
        print('No scProteomics data input')
        return {}

    root = Path(scProteomics_path)
    if not root.exists():
        raise FileNotFoundError(f"Cannot find scProteomics file: {root}")
    if not root.is_file():
        raise ValueError(f"Expected a scProteomics file, but received a directory: {root}")

    return read_celltype_feature_score_file(root, "protein")


def read_scatacseq_inputs(scATACseq_path):
    if scATACseq_path is None:
        print('No scATAC-seq data input')
        return {}

    root = Path(scATACseq_path)
    if not root.exists():
        raise FileNotFoundError(f"Cannot find scATAC-seq file: {root}")
    if not root.is_file():
        raise ValueError(f"Expected a scATAC-seq file, but received a directory: {root}")

    return read_celltype_feature_score_file(root, "TF")


def format_pathway_line(pathway):
    columns = list(pathway)
    if len(columns) == 6:
        columns[-1] = f"{float(columns[-1]):.10g}"
    return "\t".join(map(str, columns)) + "\n"


def has_positive_multiomics_match(
        pathways,
        sender_cell,
        receiver_cell,
        dict_proteomics,
        dict_atacseq
    ):
    sender_proteins = dict_proteomics.get(sender_cell, {})
    receiver_proteins = dict_proteomics.get(receiver_cell, {})
    receiver_tfs = dict_atacseq.get(receiver_cell, {})

    for pathway in pathways:
        if any(
            sender_proteins.get(ligand, 0.0) > 0
            for ligand in pathway[0].split(";")
            if ligand
        ):
            return True
        if receiver_proteins.get(pathway[1], 0.0) > 0:
            return True
        if receiver_tfs.get(pathway[3], 0.0) > 0:
            return True

    return False


def save_inferred_pathways(
        pathways,
        sender_cell,
        receiver_cell,
        output_dir,
        dict_proteomics,
        dict_atacseq,
        ko_weight,
        ligand_weight,
        receptor_weight,
        tf_weight,
        multiomics_output_dir=None
    ):
    os.makedirs(output_dir, exist_ok=True)
    safe_sender = safe_filename_component(sender_cell)
    safe_receiver = safe_filename_component(receiver_cell)
    pathway_filename = f'{safe_sender}_to_{safe_receiver}_pathway.txt'
    output_file = os.path.join(output_dir, pathway_filename)

    has_pair_multiomics = (
        (
            sender_cell in dict_proteomics
            or receiver_cell in dict_proteomics
            or receiver_cell in dict_atacseq
        )
        and has_positive_multiomics_match(
            pathways=pathways,
            sender_cell=sender_cell,
            receiver_cell=receiver_cell,
            dict_proteomics=dict_proteomics,
            dict_atacseq=dict_atacseq
        )
    )
    pathways = sorted(pathways, key=lambda pathway: pathway[-1], reverse=True)

    with open(output_file, 'w', encoding='utf-8', newline='') as handle:
        handle.write("Ligand\tReceptor\tMediator\tTF\tTarget\tEvidence_score\n")
        for pathway in pathways:
            handle.write(format_pathway_line(pathway))

    if multiomics_output_dir is not None and has_pair_multiomics:
        multiomics_pathways = []
        for pathway in pathways:
            multiomics_pathways.extend(
                score_pathway_with_multiomics(
                    pathway=pathway,
                    sender_cell=sender_cell,
                    receiver_cell=receiver_cell,
                    dict_proteomics=dict_proteomics,
                    dict_atacseq=dict_atacseq,
                    ko_weight=ko_weight,
                    ligand_weight=ligand_weight,
                    receptor_weight=receptor_weight,
                    tf_weight=tf_weight
                )
            )
        multiomics_pathways.sort(key=lambda pathway: pathway[-1], reverse=True)
        os.makedirs(multiomics_output_dir, exist_ok=True)
        supported_output_file = os.path.join(
            multiomics_output_dir,
            pathway_filename
        )
        with open(supported_output_file, 'w', encoding='utf-8', newline='') as handle:
            handle.write("Ligand\tReceptor\tMediator\tTF\tTarget\tEvidence_score\n")
            for pathway in multiomics_pathways:
                handle.write(format_pathway_line(pathway))


def score_pathway_with_multiomics(
        pathway,
        sender_cell,
        receiver_cell,
        dict_proteomics,
        dict_atacseq,
        ko_weight,
        ligand_weight,
        receptor_weight,
        tf_weight
    ):
    sender_proteins = dict_proteomics.get(sender_cell)
    ligands = [ligand for ligand in pathway[0].split(";") if ligand]

    if sender_proteins is None:
        ligand_variants = [(pathway[0], None)]
    else:
        positive_ligands = [
            (ligand, sender_proteins.get(ligand, 0.0))
            for ligand in ligands
            if sender_proteins.get(ligand, 0.0) > 0
        ]
        unsupported_ligands = [
            ligand
            for ligand in ligands
            if sender_proteins.get(ligand, 0.0) <= 0
        ]
        ligand_variants = positive_ligands
        if unsupported_ligands:
            ligand_variants.append((";".join(unsupported_ligands), 0.0))
        if not ligand_variants:
            ligand_variants = [(pathway[0], 0.0)]

    receiver_proteins = dict_proteomics.get(receiver_cell)
    receiver_tfs = dict_atacseq.get(receiver_cell)
    scored_pathways = []

    for ligand_names, ligand_score in ligand_variants:
        weighted_score = ko_weight * float(pathway[-1])
        available_weight = ko_weight

        if ligand_score is not None:
            weighted_score += ligand_weight * ligand_score
            available_weight += ligand_weight

        if receiver_proteins is not None:
            receptor_score = receiver_proteins.get(pathway[1], 0.0)
            weighted_score += receptor_weight * receptor_score
            available_weight += receptor_weight

        if receiver_tfs is not None:
            tf_score = receiver_tfs.get(pathway[3], 0.0)
            weighted_score += tf_weight * tf_score
            available_weight += tf_weight

        scored_pathway = list(pathway)
        scored_pathway[0] = ligand_names
        scored_pathway[-1] = weighted_score / available_weight
        scored_pathways.append(scored_pathway)

    return scored_pathways


parser = argparse.ArgumentParser(description='Main entrance of SigFormer')
parser.add_argument('--scRNAseq_path', type=str, default=None,
                    help='the path of scRNA-seq data')
parser.add_argument('--scProteomics_path', type=str, default=None,
                    help='tab-delimited scProteomics file: celltype, protein, optional Score')
parser.add_argument('--scATACseq_path', type=str, default=None,
                    help='tab-delimited scATAC-seq file: celltype, TF, optional Score')
parser.add_argument('--pathway_file', type=str, default='./reference_library/Intracellular_signaling.txt',
                    help='the path of curated intracellular signaling interactions')
parser.add_argument('--ligand_file', type=str, default='./reference_library/Ligand_secreted_and_membrane.txt',
                    help='curated ligand-receptor pairs used when spatial=false')
parser.add_argument('--membrane_ligand_file', type=str, default='./reference_library/Ligand_membrane.txt',
                    help='membrane-bound ligand-receptor pairs used when spatial=true')
parser.add_argument('--secreted_ligand_file', type=str, default='./reference_library/Ligand_secreted.txt',
                    help='secreted ligand-receptor pairs used when spatial=true')
parser.add_argument('--index_cell', type=str, default='Malignant',
                    help='the index cell type')
parser.add_argument('--retained_cell_types', type=str, default=None,
                    help='optional cell-type file (one exact label per line or first TSV/CSV column); filters cells; empty retains all annotated types')
parser.add_argument('--min_cell', type=float, default=0.01,
                    help='parameter for gene filtering')
parser.add_argument('--min_gene', type=float, default=0.01,
                    help='parameter for cell filtering')
parser.add_argument('--normalize', type=str2bool, nargs='?', const=True, default=True,
                    help='normalize cells; accepts true/false, yes/no, or 1/0')
parser.add_argument('--log_trans', type=str2bool, nargs='?', const=True, default=True,
                    help='logarithm expression; accepts true/false, yes/no, or 1/0')
parser.add_argument('--hvg_top_gene', type=int, default=5000,
                    help='take the top X highly variable genes when background_gene_set is not provided (default: 5000)')
parser.add_argument('--background_gene_set', type=str, default=None,
                    help='optional TXT/TSV/CSV file whose first column contains background genes; overrides HVG selection')
parser.add_argument('--cell_top_gene', type=int, default=500,
                    help='take the top X expressed genes within the selected background for each cell (default: 500)')
parser.add_argument('--spatial', type=str2bool, nargs='?', const=True, default=False,
                    help='using spatial information; accepts true/false, yes/no, or 1/0')
parser.add_argument('--knn', type=int, default=10,
                    help='number of nearest spatial neighbors per cell before excluding same-cell-type neighbors')
parser.add_argument('--spatial_ligand_min_fraction', type=float, default=0.1,
                    help='minimum sender-cell expression fraction for spatial ligands; selection is strictly greater')
parser.add_argument('--classification_accuracy', type=float, default=0.8, help='Threshold for cell classification')
parser.add_argument('--edge_threshold', type=float, default=0.8, help='Threshold for edge reconstruction')
parser.add_argument('--ko_evidence_weight', type=float, default=1.0,
                    help='weight of the receptor-to-TF perturbation score in pathway scoring')
parser.add_argument('--ligand_evidence_weight', type=float, default=1.0,
                    help='weight of sender ligand protein abundance in pathway scoring')
parser.add_argument('--receptor_evidence_weight', type=float, default=1.0,
                    help='weight of receiver receptor protein abundance in pathway scoring')
parser.add_argument('--tf_evidence_weight', type=float, default=1.0,
                    help='weight of receiver TF chromatin-binding potential in pathway scoring')
parser.add_argument('--min_cell_count', type=int, default=5, help='Minimum cell count for each cell type')
parser.add_argument('--num_epochs', type=int, default=50, help='training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for model optimization')
parser.add_argument('--block_size', type=int, default=5000, help='size of each segment block')
parser.add_argument('--metacell_expr_threshold', type=float, default=0.05,
                    help='set metacell mean expression values below this threshold to 0; use 0 to disable')
parser.add_argument('--random_seed', type=int, default=43, help='random seed')
args = parser.parse_args()

if not 0 <= args.min_cell <= 1:
    parser.error("min_cell must be between 0 and 1")
if not 0 <= args.min_gene <= 1:
    parser.error("min_gene must be between 0 and 1")
if args.hvg_top_gene < 1:
    parser.error("hvg_top_gene must be at least 1")
if args.cell_top_gene < 1:
    parser.error("cell_top_gene must be at least 1")
if args.knn < 1:
    parser.error("knn must be at least 1")
if not 0 <= args.spatial_ligand_min_fraction <= 1:
    parser.error("spatial_ligand_min_fraction must be between 0 and 1")
if not 0 <= args.classification_accuracy <= 1:
    parser.error("classification_accuracy must be between 0 and 1")
if not 0 <= args.edge_threshold <= 1:
    parser.error("edge_threshold must be between 0 and 1")
if args.min_cell_count < 1:
    parser.error("min_cell_count must be at least 1")
if args.num_epochs < 1:
    parser.error("num_epochs must be at least 1")
if args.learning_rate <= 0:
    parser.error("learning_rate must be positive")
if args.block_size < 1:
    parser.error("block_size must be at least 1")
if args.metacell_expr_threshold < 0:
    parser.error("metacell_expr_threshold must be non-negative")
if args.ko_evidence_weight <= 0:
    parser.error("ko_evidence_weight must be positive")
if any(weight < 0 for weight in (
    args.ligand_evidence_weight,
    args.receptor_evidence_weight,
    args.tf_evidence_weight
)):
    parser.error("Multi-omics evidence weights must be non-negative")

if args.retained_cell_types is not None and not args.retained_cell_types.strip():
    args.retained_cell_types = None

retained_cell_types = None
if args.retained_cell_types is not None:
    retained_cell_types_path = Path(args.retained_cell_types)
    if not retained_cell_types_path.is_file():
        parser.error(f"Cannot find the retained cell types file: {args.retained_cell_types}")
    try:
        retained_cell_types = read_retained_cell_types(retained_cell_types_path)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.index_cell not in retained_cell_types:
        parser.error(f"retained_cell_types must include index_cell '{args.index_cell}'")
    if len(retained_cell_types) < 2:
        parser.error("retained_cell_types must contain at least two distinct cell types")
    print(f"Requested cell types: {', '.join(retained_cell_types)}")

if args.background_gene_set is not None and not args.background_gene_set.strip():
    args.background_gene_set = None

background_genes = None
if args.background_gene_set is not None:
    background_gene_path = Path(args.background_gene_set)
    if not background_gene_path.is_file():
        parser.error(f"Cannot find the background gene set file: {args.background_gene_set}")
    try:
        background_genes = read_background_gene_set(background_gene_path)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Loaded {len(background_genes)} background genes from "
        f"{background_gene_path}."
    )

if args.scRNAseq_path is None or not Path(args.scRNAseq_path).exists():
    print(f"Cannot find the scRNA-seq folder or file: {args.scRNAseq_path}")
    sys.exit(1)

if args.pathway_file is None or not Path(args.pathway_file).is_file():
    print(f"Cannot find the reference pathway file: {args.pathway_file}")
    sys.exit(1)

if args.spatial:
    for ligand_file in (args.membrane_ligand_file, args.secreted_ligand_file):
        if ligand_file is None or not Path(ligand_file).is_file():
            print(f"Cannot find the spatial ligand-receptor file: {ligand_file}")
            sys.exit(1)
elif args.ligand_file is None or not Path(args.ligand_file).is_file():
    print(f"Cannot find the reference ligand-receptor file: {args.ligand_file}")
    sys.exit(1)

try:
    dict_Proteomics = read_scproteomics_inputs(args.scProteomics_path)
    dict_ATACseq = read_scatacseq_inputs(args.scATACseq_path)
except (OSError, ValueError) as exc:
    parser.error(f"Error reading optional multi-omics input: {exc}")

root = Path(args.scRNAseq_path)
if root.is_file():
    if root.suffix.lower() != ".h5ad":
        parser.error(f"scRNAseq_path must be an .h5ad file or a directory: {root}")
    file_list = [root]
else:
    file_list = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".h5ad"
        ),
        key=lambda path: str(path).casefold()
    )

if not file_list:
    parser.error(f"No .h5ad files were found under scRNAseq_path: {root}")

files_by_stem = {}
for input_file in file_list:
    files_by_stem.setdefault(input_file.stem.casefold(), []).append(input_file)
duplicate_stems = {
    stem: paths
    for stem, paths in files_by_stem.items()
    if len(paths) > 1
}
if duplicate_stems:
    duplicate_description = "; ".join(
        f"{stem}: {', '.join(map(str, paths))}"
        for stem, paths in sorted(duplicate_stems.items())
    )
    parser.error(
        "Input .h5ad files must have unique filename stems because each stem is used "
        f"as its result directory name. Duplicates: {duplicate_description}"
    )

for file in file_list:
    if file.suffix.lower() == ".h5ad":
        start = time.time()
        adata = read_file(
            file_path=file, index_cell=args.index_cell,
            retained_cell_types=retained_cell_types,
        )
        label_sample = adata.obs['sample'].unique()
        all_perturbation_results = {}

        spatial_obsm_key = None
        if args.spatial:
            spatial_obsm_key = get_spatial_obsm_key(adata)
            if spatial_obsm_key is None:
                print(
                    f"When 'spatial' is True, {file.stem}.h5ad must have a numeric spatial-like "
                    f".obsm entry such as 'spatial' or 'spatial_embeddings'"
                )
                sys.exit(1)
            print(f"Use adata.obsm['{spatial_obsm_key}'] as spatial coordinates.")

        dict_all = {}
        dict_gene = {}
        rejected_sample = 0
        spatial_neighbor_sender_types = {}
        spatial_distant_sender_types = {}
        spatial_ligand_expression_counts = {
            "membrane": {},
            "secreted": {}
        }
        for sample in label_sample:
            if sample != '':
                sub_adata = adata[adata.obs['sample'] == sample].copy()
                sub_adata = sub_adata[(sub_adata.obs['celltype'].notna()) & (sub_adata.obs['celltype'] != '')].copy()

                print(f"Initial size of {sample}: {sub_adata.shape}")
                sub_adata = preprocess(sub_adata, args.hvg_top_gene, background_genes=background_genes,
                                       min_cell=args.min_cell, min_gene=args.min_gene,
                                       normalize=args.normalize, log_trans=args.log_trans)
                if sub_adata is None:
                    continue
                if sub_adata.n_vars < args.cell_top_gene:
                    print(
                        f"Warning: sample {sample} contains only {sub_adata.n_vars} background genes, "
                        f"fewer than cell_top_gene={args.cell_top_gene}. All expressed background "
                        "genes will be retained for each cell."
                    )

                # remove cell types with fewer than 5 cell
                ct_counts = sub_adata.obs["celltype"].value_counts()
                keep_types = ct_counts[ct_counts >= args.min_cell_count].index.tolist()
                if len(keep_types) == 0:
                    continue  # Skip to the next iteration of the loop
                sub_adata = sub_adata[sub_adata.obs["celltype"].isin(keep_types)].copy()

                print(f"processed size of {sample}: {sub_adata.shape}")

                print('Read pathway file...')
                gene_list = sub_adata.var_names
                gene_dict = {var: idx for idx, var in enumerate(gene_list)}
                pathway_matrix, pathway_type, tf_dict = pathway2(args.pathway_file, gene_dict, col1_index=0,
                                                                 col2_index=1, col3_index=2)

                if args.spatial:
                    membrane_lg_set, membrane_rp_set, _ = ligand_receptor(
                        args.membrane_ligand_file, gene_dict, 0, 1, 2, 2
                    )
                    secreted_lg_set, secreted_rp_set, _ = ligand_receptor(
                        args.secreted_ligand_file, gene_dict, 0, 1, 2, 2
                    )
                    rp_set = membrane_rp_set | secreted_rp_set
                else:
                    _, rp_set, _ = ligand_receptor(args.ligand_file, gene_dict, 0, 1, 2, 2)

                unique_celltypes = sub_adata.obs['celltype'].unique()
                cellid_count = len(unique_celltypes)
                index_cell_count = (sub_adata.obs['celltype'] == args.index_cell).sum()
                if index_cell_count < args.min_cell_count or cellid_count < 2:
                    print(f"Skipping sample {sample} due to too few {args.index_cell} cells")
                    continue  # Skip to the next iteration of the loop

                sample_neighbor_sender_types = {}
                sample_distant_sender_types = {}
                sample_ligand_expression_counts = {
                    "membrane": {},
                    "secreted": {}
                }
                if args.spatial:
                    (
                        neighbor_indices_by_celltype,
                        cell_knn_indices
                    ) = find_celltype_knn_indices(
                        sub_adata,
                        embedding_key=spatial_obsm_key,
                        celltype_key="celltype",
                        n_neighbors=args.knn,
                        metric='euclidean'
                    )
                    (
                        sample_neighbor_sender_types,
                        sample_distant_sender_types,
                        distant_indices_by_celltype
                    ) = get_spatial_sender_type_sets(
                        sub_adata,
                        neighbor_indices_by_celltype,
                        cell_knn_indices,
                        celltype_key="celltype"
                    )
                    sample_ligand_expression_counts["membrane"] = (
                        collect_spatial_ligand_expression_counts(
                            sub_adata,
                            neighbor_indices_by_celltype,
                            membrane_lg_set,
                            celltype_key="celltype"
                        )
                    )
                    sample_ligand_expression_counts["secreted"] = (
                        collect_spatial_ligand_expression_counts(
                            sub_adata,
                            distant_indices_by_celltype,
                            secreted_lg_set,
                            celltype_key="celltype"
                        )
                    )

                list_adata = [compress_anndata(
                    sub_adata,
                    args.block_size,
                    expression_threshold=args.metacell_expr_threshold,
                    clustering_obsm_key=(spatial_obsm_key if args.spatial else None)
                )]

                mean_sum = {}
                mean_count = {}
                accepted_subdata_count = 0
                for subdata in list_adata:
                    # remove cell types with fewer than min_cell_count cells/metacells after compression
                    ct_counts = subdata.obs["celltype"].value_counts()
                    keep_types = ct_counts[ct_counts >= args.min_cell_count].index.tolist()
                    if len(keep_types) == 0:
                        continue  # Skip to the next iteration of the loop
                    subdata = subdata[subdata.obs["celltype"].isin(keep_types)].copy()

                    unique_celltypes = subdata.obs['celltype'].unique()
                    if args.index_cell not in unique_celltypes or len(unique_celltypes) < 2:
                        print(f"Skipping sample {sample} due to too few {args.index_cell} cells")
                        continue  # Skip to the next iteration of the loop

                    cellid_count = len(unique_celltypes)
                    count_dict = Counter(subdata.obs['celltype'])
                    print(count_dict)

                    label_id = {label: idx for idx, label in enumerate(unique_celltypes)}
                    id_label = {idx: label for label, idx in label_id.items()}
                    cell_id = subdata.obs['celltype'].map(label_id)
                    subdata.obs['cellid'] = cell_id

                    data = subdata.X.toarray() if sp.issparse(subdata.X) else subdata.X
                    activated_genes = []
                    for row in data:
                        expressed_gene_indices = np.where(row > 0)[0]
                        expressed_values = row[expressed_gene_indices]

                        if len(expressed_gene_indices) <= args.cell_top_gene:
                            activated_genes.append(expressed_gene_indices)
                        else:
                            sorted_values = np.sort(expressed_values)[::-1]  # Descending
                            cutoff = sorted_values[args.cell_top_gene - 1]
                            selected_indices = expressed_gene_indices[expressed_values >= cutoff]
                            act_genes = set(selected_indices)

                            enriched_tfs = []
                            for tf, targets in tf_dict.items():
                                overlap_genes = act_genes.intersection(
                                    targets)  # Number of activated genes that are TF targets
                                if len(overlap_genes) > 0 and row[tf] > 0:
                                    enriched_tfs.append(tf)
                            enriched_tfs = np.asarray(enriched_tfs, dtype=int)
                            selected_indices = selected_indices.astype(int, copy=False)
                            merged = np.unique(np.concatenate([enriched_tfs, selected_indices]))
                            activated_genes.append(merged)

                    data_list = csnet(
                        subdata,
                        activated_genes,
                        pathway_matrix,
                        pathway_type=pathway_type
                    )
                    accuracy, adj_dict, perturbation_results = graph_processing(
                        data_list=data_list,
                        cell_idx=range(len(data_list)),
                        gene_names=gene_list,
                        cell_labels=id_label,
                        out_dim=cellid_count,
                        num_epochs=args.num_epochs,
                        learning_rate=args.learning_rate,
                        min_cell_count=args.min_cell_count,
                        seed=args.random_seed,
                        knockout_gene_idx=rp_set,
                        classification_accuracy=args.classification_accuracy
                    )
                    index_label = label_id[args.index_cell]
                    has_index_network = index_label in adj_dict
                    has_partner_network = any(
                        label != index_label
                        for label in adj_dict
                    )
                    if (
                        accuracy < args.classification_accuracy
                        or not has_index_network
                        or not has_partner_network
                    ):
                        rejected_sample += 1
                        if accuracy < args.classification_accuracy:
                            reason = (
                                f"classification accuracy {accuracy:.4f} is below "
                                f"{args.classification_accuracy:.4f}"
                            )
                        elif not has_index_network:
                            reason = (
                                f"fewer than {args.min_cell_count} correctly classified "
                                f"{args.index_cell} cells were available for network reconstruction"
                            )
                        else:
                            reason = (
                                "no non-index cell type had enough correctly classified cells "
                                "for network reconstruction"
                            )
                        print(f"Reject sample {sample}: {reason}.")
                        del adj_dict
                        del perturbation_results
                        del data_list
                        del subdata
                        del data
                        del activated_genes
                        continue

                    for cell_type, perturb in perturbation_results.items():
                        if cell_type not in all_perturbation_results:
                            all_perturbation_results[cell_type] = {}
                        for ko_gene, perturb_genes in perturb.items():
                            if ko_gene not in all_perturbation_results[cell_type]:
                                all_perturbation_results[cell_type][ko_gene] = []
                            all_perturbation_results[cell_type][ko_gene].append(perturb_genes)
                    del perturbation_results

                    accepted_subdata_count += 1
                    for key, matrix in adj_dict.items():
                        cell_name = id_label[key]
                        if cell_name not in mean_sum:
                            mean_sum[cell_name] = matrix.copy()  # Initialize with the first matrix
                            mean_count[cell_name] = 1
                        else:
                            mean_sum[cell_name] += matrix  # Add the matrix to the sum
                            mean_count[cell_name] += 1  # Increment the count
                    del adj_dict

                    del data_list
                    del subdata
                    del data
                    del activated_genes
                    gc.collect()

                adj_dict_mean = {}
                if accepted_subdata_count > 0:
                    if args.spatial:
                        for receiver_celltype, sender_types in sample_neighbor_sender_types.items():
                            spatial_neighbor_sender_types.setdefault(receiver_celltype, set()).update(sender_types)
                        for receiver_celltype, sender_types in sample_distant_sender_types.items():
                            spatial_distant_sender_types.setdefault(receiver_celltype, set()).update(sender_types)
                        for mode_name in ("membrane", "secreted"):
                            merge_ligand_expression_counts(
                                spatial_ligand_expression_counts[mode_name],
                                sample_ligand_expression_counts[mode_name]
                            )

                    adj_dict_mean = {key: (mean_sum[key] / mean_count[key]).tocoo() for key in mean_sum}
                    for id, adj_mean in adj_dict_mean.items():
                        mask = adj_mean.data >= args.edge_threshold
                        rows = adj_mean.row[mask]
                        cols = adj_mean.col[mask]
                        data = adj_mean.data[mask]  # Keep the GAE edge weights instead of forcing them to 1.
                        adj_matrix = coo_matrix((data, (rows, cols)), shape=adj_mean.shape)

                        if id in dict_all:
                            dict_all[id].append(adj_matrix)
                            dict_gene[id].append(gene_list)
                        else:
                            dict_all[id] = [adj_matrix]
                            dict_gene[id] = [gene_list]
                else:
                    print(f"Reject sample {sample}: no subdata passed classification accuracy threshold.")

                del mean_sum
                del mean_count
                del adj_dict_mean
                del sub_adata
                del list_adata
                del pathway_matrix
                del pathway_type
                del tf_dict
                del gene_list
                del gene_dict
                gc.collect()

        if len(dict_all) == 0:
            if rejected_sample > 0:
                print("All samples have been discarded. Please lower the threshold for "
                      "classification accuracy, or adjust the model training epochs or learning rate.")
            else:
                print("All samples have been discarded. Please check the input data.")
            sys.exit(1)

        ########################################## Pathway Reconstruction ####################################################
        sum_matrix = {}
        sum_gene_lists = {}
        for cell, adj_list in dict_all.items():
            gene_lists = dict_gene[cell]
            adj_integrated, gene_integrated = integrate_multiple_graphs(cell, adj_list, gene_lists, len(adj_list))
            sum_matrix[cell] = adj_integrated
            sum_gene_lists[cell] = gene_integrated

        final_matrix, unified_gene_list = align_adjacency_matrices(sum_matrix, sum_gene_lists)
        unified_gene_dict = {var: idx for idx, var in enumerate(unified_gene_list)}

        if args.spatial:
            spatial_ligands = {
                mode_name: select_spatial_ligands(
                    spatial_ligand_expression_counts[mode_name],
                    unified_gene_dict,
                    args.spatial_ligand_min_fraction
                )
                for mode_name in ("membrane", "secreted")
            }

        recon_dir0 = f'./result/{file.stem}/cell_network'
        save_cell_specific_networks(final_matrix, unified_gene_list, recon_dir0)

        all_perturbation_results = integrate_multiple_dicts(all_perturbation_results)
        for key1 in list(all_perturbation_results.keys()):
            new_inner = {}
            for key2, gene_list in all_perturbation_results[key1].items():
                if key2 in unified_gene_dict:
                    ko_index = unified_gene_dict[key2]
                    gene_scores = {
                        unified_gene_dict[g]: score
                        for g, score in gene_list.items()
                        if g in unified_gene_dict
                    }
                    if len(gene_scores) > 0:
                        new_inner[ko_index] = gene_scores
            if len(new_inner) > 0:
                all_perturbation_results[key1] = new_inner
            else:
                del all_perturbation_results[key1]

        '''
        for cell_type, perturbation in all_perturbation_results.items():
            for ko_gene, sig_genes in perturbation.items():
                print(f"{cell_type}\t{ko_gene}\t{len(sig_genes)}")
        '''

        pathway_matrix, pathway_type, tf_dict = pathway2(args.pathway_file, unified_gene_dict, col1_index=0,
                                                         col2_index=1, col3_index=2)
        if args.spatial:
            _, _, membrane_lgrp_dict = ligand_receptor(
                args.membrane_ligand_file,
                unified_gene_dict,
                col1_index=0,
                col2_index=1,
                col3_index=2,
                min_score=2
            )
            _, _, secreted_lgrp_dict = ligand_receptor(
                args.secreted_ligand_file,
                unified_gene_dict,
                col1_index=0,
                col2_index=1,
                col3_index=2,
                min_score=2
            )
        else:
            _, _, lgrp_dict = ligand_receptor(
                args.ligand_file,
                unified_gene_dict,
                col1_index=0,
                col2_index=1,
                col3_index=2,
                min_score=2
            )

        dict_gene = {}
        for cell, adj_cell in final_matrix.items():
            gene_set = set()
            for r, c, w in zip(adj_cell.row, adj_cell.col, adj_cell.data):
                gene_set.add(int(r))
                gene_set.add(int(c))
            dict_gene[cell] = gene_set

        # divide cell networks into an tf_tg networks (tftg_final) and an extranuclear signaling networks (pathway_final)
        tftg_final = {}
        pathway_final = {}
        for cell, adj_sum in final_matrix.items():
            adj_tftg, adj_pathway = split_adjacency_matrix(adj_sum, pathway_type)
            tftg_final[cell] = adj_tftg
            pathway_final[cell] = adj_pathway

        inferred_root = f'./result/{file.stem}/scRNAseq-inferred'
        multiomics_root = f'./result/{file.stem}/multiomics-supported'
        has_multiomics = (
            bool(set(dict_gene) & set(dict_Proteomics))
            or bool(set(dict_gene) & set(dict_ATACseq))
        )

        if args.spatial:
            inferred_dirs = {
                "membrane": os.path.join(inferred_root, "membrane-mediated"),
                "secreted": os.path.join(inferred_root, "secreted-mediated")
            }
            multiomics_dirs = {
                "membrane": os.path.join(multiomics_root, "membrane-mediated"),
                "secreted": os.path.join(multiomics_root, "secreted-mediated")
            }
            for output_dir in inferred_dirs.values():
                os.makedirs(output_dir, exist_ok=True)
            if has_multiomics:
                for output_dir in multiomics_dirs.values():
                    os.makedirs(output_dir, exist_ok=True)
        else:
            inferred_dirs = {"combined": inferred_root}
            multiomics_dirs = {"combined": multiomics_root}
            os.makedirs(inferred_root, exist_ok=True)
            if has_multiomics:
                os.makedirs(multiomics_root, exist_ok=True)

        if args.index_cell not in all_perturbation_results:
            print(f"No significantly perturbed genes identified in {args.index_cell}")
            sys.exit(1)

        pathway_pairs = []
        for cell in dict_gene:
            if cell == args.index_cell:
                continue
            pathway_pairs.append((cell, args.index_cell))
            if cell in pathway_final and cell in tftg_final and cell in all_perturbation_results:
                pathway_pairs.append((args.index_cell, cell))

        inference_batches = []
        for sender_cell, receiver_cell in pathway_pairs:
            if args.spatial:
                pathway_modes = []
                if sender_cell in spatial_neighbor_sender_types.get(receiver_cell, set()):
                    membrane_ligands = spatial_ligands["membrane"].get(
                        (sender_cell, receiver_cell),
                        set()
                    )
                    if len(membrane_ligands) > 0:
                        pathway_modes.append(
                            ("membrane", membrane_lgrp_dict, membrane_ligands)
                        )
                if sender_cell in spatial_distant_sender_types.get(receiver_cell, set()):
                    secreted_ligands = spatial_ligands["secreted"].get(
                        (sender_cell, receiver_cell),
                        set()
                    )
                    if len(secreted_ligands) > 0:
                        pathway_modes.append(
                            ("secreted", secreted_lgrp_dict, secreted_ligands)
                        )
            else:
                pathway_modes = [("combined", lgrp_dict, dict_gene[sender_cell])]

            if len(pathway_modes) == 0:
                print(
                    f"Skip {sender_cell} to {receiver_cell}: no spatial ligand passed "
                    f"the >{args.spatial_ligand_min_fraction:.1%} expression threshold."
                )
                continue

            for mode_name, mode_lgrp_dict, mode_ligands in pathway_modes:
                mode_description = f" {mode_name}-mediated" if args.spatial else ""
                print(f"Infer{mode_description} {sender_cell} to {receiver_cell} pathway...")
                inferred_pathways = infer_pathway(
                    unified_gene_list,
                    mode_ligands,
                    all_perturbation_results[receiver_cell],
                    mode_lgrp_dict,
                    pathway_final[receiver_cell],
                    tftg_final[receiver_cell]
                )

                if len(inferred_pathways) == 0:
                    continue

                print(
                    f"Found {len(inferred_pathways)}{mode_description} pathways "
                    f"from {sender_cell} to {receiver_cell}"
                )
                inference_batches.append(
                    {
                        "pathways": inferred_pathways,
                        "sender_cell": sender_cell,
                        "receiver_cell": receiver_cell,
                        "output_dir": inferred_dirs[mode_name],
                        "multiomics_output_dir": (
                            multiomics_dirs[mode_name] if has_multiomics else None
                        )
                    }
                )

        normalize_inferred_pathway_ko_scores(inference_batches)
        for batch in inference_batches:
            save_inferred_pathways(
                batch["pathways"],
                batch["sender_cell"],
                batch["receiver_cell"],
                batch["output_dir"],
                dict_Proteomics,
                dict_ATACseq,
                ko_weight=args.ko_evidence_weight,
                ligand_weight=args.ligand_evidence_weight,
                receptor_weight=args.receptor_evidence_weight,
                tf_weight=args.tf_evidence_weight,
                multiomics_output_dir=batch["multiomics_output_dir"]
            )
        end = time.time()
        print(f'{file} signaling pathway inference done with {end - start:.4f} s')
