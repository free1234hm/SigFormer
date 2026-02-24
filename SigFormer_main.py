import os
import sys
import argparse
import gc
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from scipy.sparse import coo_matrix
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from csn.GAE import graph_processing, graph_processing_spatial
from csn.cell_net import csnet
from read_data.read_interaction import pathway2, ligand_receptor
from csn.Integrate_graphs import integrate_multiple_graphs
from csn.Align_matrices import align_adjacency_matrices
from csn.Infer_pathway import infer_pathway


def read_file(file_path, index_cell):
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
    if index_cell not in adata.obs['celltype'].values:
        print(f"Warning: adata.obs['celltype'] must contain '{index_cell}'")
        sys.exit(1)

    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    return adata


def preprocess(adata,
               hvg_top_genes,
               min_cell=0.01,
               min_gene=0.01,
               normalize: bool = True,
               log_trans: bool = True
               ):
    try:
        # Filter cells
        min_genes = max(1, int(min_gene * adata.n_vars))
        sc.pp.filter_cells(adata, min_genes=min_genes)

        # Filter genes
        min_cells = max(1, int(min_cell * adata.n_obs))
        sc.pp.filter_genes(adata, min_cells=min_cells)

        # Normalize and Log Transformation
        X = adata.X
        max_val = X.max() if hasattr(X, "max") else np.max(X)
        all_integer = np.all(np.mod(X.data if hasattr(X, "data") else X.ravel(), 1) == 0)
        logtransformed = (max_val < 100) and not all_integer

        # detect normalization if raw counts are stored
        raw_exists = hasattr(adata, "layers") and "counts" in adata.layers
        if raw_exists:
            raw = adata.layers["counts"]
            cell_sums_raw = np.asarray(raw.sum(axis=1)).ravel()
            normalized_previously = False  # raw exists implies normalization not applied to raw
        else:
            # if raw is not stored, we check per-cell totals in current adata
            cell_sums = np.asarray(adata.X.sum(axis=1)).ravel()
            normalized_previously = np.allclose(cell_sums, cell_sums[0], atol=1e-6)

        if normalize and not normalized_previously and not logtransformed:
            print("Run cell normalization")
            sc.pp.normalize_total(adata)
        if log_trans and not logtransformed:
            print("Run log transformation")
            sc.pp.log1p(adata)

        '''
        if "spatial" in adata.obsm:
            coords = adata.obsm["spatial"]
            scaler = MinMaxScaler()
            normalized_coords = scaler.fit_transform(coords)
            adata.obsm["spatial"] = normalized_coords
        '''
        if adata.n_vars > hvg_top_genes:
            sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top_genes)
            hvg_genes = adata.var[adata.var['highly_variable']].index
            filter_adata = adata[:, list(hvg_genes)]
            return filter_adata
        else:
            return adata
    except Exception as e:
        print('Error in identifying highly variable genes')


def split_anndata(adata, block_size):
    n_cells = adata.n_obs

    if n_cells <= block_size:
        return [adata]
    else:
        num_splits = n_cells // block_size + 1
    print(f"Split anndata into {num_splits} parts.")

    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]

        # 3) 用 k-means 进行空间聚类
        kmeans = KMeans(n_clusters=num_splits, random_state=42)
        spatial_labels = kmeans.fit_predict(coords)

        # 4) 把聚类标签写入 obs
        adata = adata.copy()
        adata.obs["spatial_cluster"] = pd.Categorical(spatial_labels.astype(str))

        print("Cell number of each part：")
        print(adata.obs["spatial_cluster"].value_counts())

        adata_splits = []
        for cluster in sorted(adata.obs["spatial_cluster"].cat.categories):
            sub = adata[adata.obs["spatial_cluster"] == cluster].copy()
            adata_splits.append(sub)
    else:
        split_indices = [[] for _ in range(num_splits)]
        for celltype, indices in adata.obs.groupby("celltype", observed=False).indices.items():
            shuffled_indices = np.random.permutation(indices)

            splits = np.array_split(shuffled_indices, num_splits)
            for i, split in enumerate(splits):
                split_indices[i].extend(split)
        adata_splits = [adata[indices].copy() for indices in split_indices]

    return adata_splits

def split_adjacency_matrix(adj_G, adj_type):
    # Prepare lists for rows, columns, and data for matrices A and B
    A_rows, A_cols, A_data = [], [], []
    B_rows, B_cols, B_data = [], [], []
    # Iterate over non-zero elements in adj_G
    # adj_G_coo = adj_G.tocoo()  # Convert to COO format for easy iteration
    for i, j, value in zip(adj_G.row, adj_G.col, adj_G.data):
        if value > 0:  # Only consider edges present in adj_G
            if 'controls-expression-of' in adj_type[i, j]:
                A_rows.append(i)
                A_cols.append(j)
                A_data.append(1)
            if adj_type[i, j] != {'controls-expression-of'}:
                B_rows.append(i)
                B_cols.append(j)
                B_data.append(1)

    # Create sparse matrices A and B from the collected data
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


def find_knn_indices(
        adata,
        embedding_key="spatial",
        celltype_key="celltype",
        index_label="Malignant",
        n_neighbors=10,
        metric='euclidean'
    ):
    # 获取 malignant 类型
    malignant_mask = adata.obs[celltype_key].values == index_label
    malignant_idx = np.where(malignant_mask)[0]

    # 获取所有非 malignant 的整数索引
    nonmalignant_idx = np.where(~malignant_mask)[0]

    # -------------------------------------
    # 3) 构建 NearestNeighbors 模型（只用非 malignant 细胞作为训练集）
    #    algorithm='auto' 会选择合适的 KDTree/BallTree/Brute 方式
    # -------------------------------------
    nonmal_coords = adata.obsm[embedding_key][nonmalignant_idx, :]
    nbrs_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric=metric)
    nbrs_model.fit(nonmal_coords)

    # -------------------------------------
    # 4) 搜索每个 malignant 细胞最近的 non-malignant neighbors
    # -------------------------------------
    selected_set = set()

    # 把所有 malignant 的整数 obs index 加入集合
    for idx in malignant_idx:
        selected_set.add(int(idx))

    # kneighbors 输入 malignant cell 的空间坐标
    mal_coords = adata.obsm[embedding_key][malignant_idx, :]

    # 找每个 malignant 最近的 10 个 non malig
    distances, neighbor_indices = nbrs_model.kneighbors(mal_coords)

    # neighbor_indices 是 nonmalignant_idx 子集的局部索引
    for i, nm_locals in enumerate(neighbor_indices):
        for local_j in nm_locals:
            # 把局部 nonmalignant index 转换回全局整数 obs index
            global_j = int(nonmalignant_idx[local_j])
            selected_set.add(global_j)

    selected_idx_list = sorted(selected_set)
    return selected_idx_list


parser = argparse.ArgumentParser(description='Main entrance of SigFormer')
parser.add_argument('--scRNAseq_path', type=str, default=None,
                    help='the path of scRNA-seq data')
parser.add_argument('--scProteomics_path', type=str, default=None,
                    help='the path of scProteomics data')
parser.add_argument('--scATACseq_path', type=str, default=None,
                    help='the path of scATAC-seq data')
parser.add_argument('--pathway_file', type=str, default='./reference library/Intracellular signaling.txt',
                    help='the path of curated intracellular signaling interactions')
parser.add_argument('--ligand_file', type=str, default='./reference library/Ligand_secreted&membrane.txt',
                    help='the path of curated ligand-receptor pairs')
parser.add_argument('--index_cell', type=str, default='Malignant',
                    help='the index cell type')
parser.add_argument('--min_cell', type=float, default=0.01,
                    help='parameter for gene filtering')
parser.add_argument('--min_gene', type=float, default=0.01,
                    help='parameter for cell filtering')
parser.add_argument('--normalize', type=bool, default=True,
                    help='normalize cells')
parser.add_argument('--log_trans', type=bool, default=True,
                    help='logarithm expression')
parser.add_argument('--hvg_top_gene', type=int, default=5000,
                    help='take the top X highly variable genes (default: 5000)')
parser.add_argument('--cell_top_gene', type=int, default=500,
                    help='take the top X genes expressed in each cell (default: 500)')
parser.add_argument('--spatial', type=bool, default=False, help='using spatial information')
parser.add_argument('--knn', type=int, default=10,
                    help='When spatial information is available, get the k nearest neighbors of each index cell')
parser.add_argument('--classification_accuracy', type=float, default=0.8, help='Threshold for cell classification')
parser.add_argument('--edge_threshold', type=float, default=0.8, help='Threshold for edge reconstruction')
parser.add_argument('--min_cell_count', type=int, default=5, help='Minimum cell count for each cell type')
parser.add_argument('--max_length', type=int, default=5, help='max pathway length')
parser.add_argument('--num_epochs', type=int, default=20, help='training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for model optimization')
parser.add_argument('--block_size', type=int, default=5000, help='size of each segment block')
parser.add_argument('--random_seed', type=int, default=43, help='random seed')
args = parser.parse_args()

# args.scRNAseq_path = r".\test data\scRNA-seq/Data_Wang2019_Glioblastoma_SF12090.h5ad"

if args.scRNAseq_path is None or not Path(args.scRNAseq_path).exists():
    print(f"Cannot find the scRNA-seq folder or file: {args.scRNAseq_path}")
    sys.exit(1)

root = Path(args.scRNAseq_path)
if root.is_file():
    file_list = [root]
else:
    file_list = list(root.rglob("*"))

for file in file_list:
    if file.suffix == ".h5ad":
        adata = read_file(file_path=file, index_cell=args.index_cell)
        label_sample = adata.obs['sample'].unique()

        if args.spatial and "spatial" not in adata.obsm:
            print(f"When 'spatial' is True, {file.stem}.h5ad must have a .obsm['spatial'] attribute")
            sys.exit(1)

        dict_all = {}
        dict_gene = {}
        rejected_sample = 0
        for sample in label_sample:
            if sample != '':
                sub_adata = adata[adata.obs['sample'] == sample].copy()
                sub_adata = sub_adata[(sub_adata.obs['celltype'].notna()) & (sub_adata.obs['celltype'] != '')].copy()

                print(f"Initial size of {sample}: {sub_adata.shape}")
                sub_adata = preprocess(sub_adata, args.hvg_top_gene, min_cell=args.min_cell, min_gene=args.min_gene,
                                       normalize=args.normalize, log_trans=args.log_trans)
                if sub_adata is None:
                    continue

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
                pathway_matrix, pathway_type, tf_dict = pathway2(args.pathway_file, gene_dict, col1_index=0, col2_index=1,
                                                                 col3_index=2)

                unique_celltypes = sub_adata.obs['celltype'].unique()
                cellid_count = len(unique_celltypes)
                index_cell_count = (sub_adata.obs['celltype'] == args.index_cell).sum()
                if index_cell_count < args.min_cell_count or cellid_count < 2:
                    print(f"Skipping sample {sample} due to too few {args.index_cell} cells")
                    continue  # Skip to the next iteration of the loop

                list_adata = split_anndata(sub_adata, args.block_size)

                accuracy_sum = 0
                mean_sum = {}
                mean_count = {}
                for subdata in list_adata:
                    # remove cell types with fewer than 5 cell
                    ct_counts = subdata.obs["celltype"].value_counts()
                    keep_types = ct_counts[ct_counts >= args.min_cell_count].index.tolist()
                    if len(keep_types) == 0:
                        continue  # Skip to the next iteration of the loop
                    subdata = subdata[subdata.obs["celltype"].isin(keep_types)].copy()

                    unique_celltypes = subdata.obs['celltype'].unique()
                    if args.index_cell not in unique_celltypes or len(unique_celltypes) < 2:
                        print(f"Skipping sample {sample} due to too few {args.index_cell} cells")
                        continue  # Skip to the next iteration of the loop

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
                            # enriched_tfs = np.array(enriched_tfs, dtype=int)
                            merged = np.unique(np.concatenate([enriched_tfs, selected_indices]))
                            activated_genes.append(merged)

                    data_list = csnet(subdata, activated_genes, pathway_matrix)

                    if args.spatial:
                        cell_neighbors = find_knn_indices(
                            subdata,
                            embedding_key="spatial",
                            celltype_key="celltype",
                            index_label=args.index_cell,
                            n_neighbors=args.knn,
                            metric='euclidean'
                        )
                        if len(cell_neighbors) > 0:
                            accuracy, adj_dict = graph_processing_spatial(data_list, subdata.obs_names, cell_neighbors,
                                                                          out_dim=cellid_count,
                                                                          num_epochs=args.num_epochs,
                                                                          learning_rate=args.learning_rate,
                                                                          min_cell_count=args.min_cell_count,
                                                                          seed=args.random_seed)
                            accuracy_sum = accuracy_sum + accuracy
                            for key, matrix in adj_dict.items():
                                cell_name = id_label[key]
                                if cell_name not in mean_sum:
                                    mean_sum[cell_name] = matrix.copy()  # Initialize with the first matrix
                                    mean_count[cell_name] = 1
                                else:
                                    mean_sum[cell_name] += matrix  # Add the matrix to the sum
                                    mean_count[cell_name] += 1  # Increment the count
                            del adj_dict
                        else:
                            print(f"No neighboring cells to {args.index_cell} were found.")
                    else:
                        accuracy, adj_dict = graph_processing(data_list, subdata.obs_names,
                                                              out_dim=cellid_count, num_epochs=args.num_epochs,
                                                              min_cell_count=args.min_cell_count, seed=args.random_seed)
                        accuracy_sum = accuracy_sum + accuracy
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

                accuracy_sum = accuracy_sum / len(list_adata)
                adj_dict_mean = {key: (mean_sum[key] / mean_count[key]).tocoo() for key in mean_sum}

                if accuracy_sum >= args.classification_accuracy:
                    for id, adj_mean in adj_dict_mean.items():
                        mask = adj_mean.data >= args.edge_threshold
                        rows = adj_mean.row[mask]
                        cols = adj_mean.col[mask]
                        data = np.ones_like(rows, dtype=adj_mean.data.dtype)
                        adj_matrix = coo_matrix((data, (rows, cols)), shape=adj_mean.shape)

                        if id in dict_all:
                            dict_all[id].append(adj_matrix)
                            dict_gene[id].append(gene_list)
                        else:
                            dict_all[id] = [adj_matrix]
                            dict_gene[id] = [gene_list]
                else:
                    rejected_sample += 1
                    print(f"Reject sample {sample} for low classification accuracy: {accuracy_sum:.4f} ")

                del mean_sum
                del mean_count
                del adj_dict_mean
                del accuracy_sum
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
        dict_Proteomics = {}
        try:
            scProteomics_path = Path(args.scProteomics_path)
            for ff in scProteomics_path.rglob("*"):
                if ff.is_file():
                    cell_name = ff.stem
                    protein_set = set()
                    with open(ff, 'r') as f:
                        line = f.readline().strip()
                        for line in f:
                            elements = line.strip().strip('"').split("\t")
                            if float(elements[1]) > 0:
                                protein_set.add(elements[0])
                    dict_Proteomics[cell_name] = protein_set
        except Exception as e:
            print(f'No scProteomics data input')

        dict_ATACseq = {}
        try:
            scATACseq_path = Path(args.scATACseq_path)
            for ff in scATACseq_path.rglob("*"):
                if ff.is_file():
                    cell_name = ff.stem
                    tf_set = set()
                    with open(ff, 'r') as f:
                        line = f.readline().strip()
                        for line in f:
                            elements = line.strip().strip('"').split("\t")
                            tf_set.add(elements[0])
                    dict_ATACseq[cell_name] = tf_set
        except Exception as e:
            print(f'No scATAC-seq data input')

        sum_matrix = {}
        sum_gene_lists = {}
        for cell, adj_list in dict_all.items():
            gene_lists = dict_gene[cell]
            adj_integrated, gene_integrated = integrate_multiple_graphs(cell, adj_list, gene_lists, len(adj_list))
            sum_matrix[cell] = adj_integrated
            sum_gene_lists[cell] = gene_integrated

        final_matrix, unified_gene_list = align_adjacency_matrices(sum_matrix, sum_gene_lists)
        unified_gene_dict = {var: idx for idx, var in enumerate(unified_gene_list)}

        '''
        Proteomics_dict = {}
        for key, str_set in dict_Proteomics.items():
            converted_set = {unified_gene_dict[s] for s in str_set if s in unified_gene_dict}
            Proteomics_dict[key] = converted_set
        scATACseq_dict = {}
        for key, str_set in dict_ATACseq.items():
            converted_set = {unified_gene_dict[s] for s in str_set if s in unified_gene_dict}
            scATACseq_dict[key] = converted_set
        '''

        pathway_matrix, pathway_type, tf_dict = pathway2(args.pathway_file, unified_gene_dict, col1_index=0, col2_index=1,
                                                         col3_index=2)
        lgrp_matrix = ligand_receptor(args.ligand_file, unified_gene_dict, 0, 1, 2, 2)

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

        # output pathway
        recon_dir1 = f'./result/{file.stem}/scRNAseq-inferred'
        os.makedirs(recon_dir1, exist_ok=True)

        if bool(set(dict_gene) & set(dict_Proteomics)) or bool(set(dict_gene) & set(dict_ATACseq)):
            recon_dir2 = f'./result/{file.stem}/multiomics-supported'
            os.makedirs(recon_dir2, exist_ok=True)

        tftg_malignant = tftg_final[args.index_cell]
        pathway_malignant = pathway_final[args.index_cell]
        malignant_gene = dict_gene[args.index_cell]
        for cell, gene_set in dict_gene.items():
            if cell != args.index_cell:
                print(f"Infer {cell} to {args.index_cell} pathway...")
                shortest_other_to_malignant = infer_pathway(unified_gene_list, gene_set, lgrp_matrix, pathway_malignant,
                                                            tftg_malignant, args.max_length)
                if len(shortest_other_to_malignant) > 0:
                    print(f"Found {len(shortest_other_to_malignant)} pathways from {cell} to {args.index_cell}")

                    if cell in dict_Proteomics or args.index_cell in dict_Proteomics or args.index_cell in dict_ATACseq:
                        sender_protein = dict_Proteomics.get(cell, unified_gene_list)
                        receiver_protein = dict_Proteomics.get(args.index_cell, unified_gene_list)
                        receiver_tf = dict_ATACseq.get(args.index_cell, unified_gene_list)

                        result_multiomics = []
                        output_file = os.path.join(recon_dir1, f'{cell}_to_{args.index_cell}_pathway.txt')
                        with open(output_file, 'w') as f:
                            f.write(f"Ligand\tReceptor\tMediator\tTF\tTarget\n")
                            for pathway in shortest_other_to_malignant:
                                col1 = pathway[0]  # Ligands
                                col2 = pathway[1]  # Receptor
                                col3 = pathway[2]  # Mediators
                                col4 = pathway[3]  # TF
                                col5 = pathway[4]  # TGs
                                line = f"{col1}\t{col2}\t{col3}\t{col4}\t{col5}\n"
                                if col1 in sender_protein and col2 in receiver_protein and col4 in receiver_tf:
                                    result_multiomics.append(line)
                                f.write(line)

                        output_file = os.path.join(recon_dir2, f'{cell}_to_{args.index_cell}_pathway.txt')
                        with open(output_file, 'w') as f:
                            f.write(f"Ligand\tReceptor\tMediator\tTF\tTarget\n")
                            for line in result_multiomics:
                                f.write(line)
                    else:
                        output_file = os.path.join(recon_dir1, f'{cell}_to_{args.index_cell}_pathway.txt')
                        with open(output_file, 'w') as f:
                            f.write(f"Ligand\tReceptor\tMediator\tTF\tTarget\n")
                            for pathway in shortest_other_to_malignant:
                                col1 = pathway[0]  # Ligands
                                col2 = pathway[1]  # Receptor
                                col3 = pathway[2]  # Mediators
                                col4 = pathway[3]  # TF
                                col5 = pathway[4]  # TGs
                                f.write(f"{col1}\t{col2}\t{col3}\t{col4}\t{col5}\n")

                if cell in pathway_final and cell in tftg_final:
                    print(f"Infer {args.index_cell} to {cell} pathway...")
                    adj_pathway = pathway_final[cell]
                    adj_tftg = tftg_final[cell]
                    shortest_malignant_to_other = infer_pathway(unified_gene_list, malignant_gene, lgrp_matrix, adj_pathway,
                                                                adj_tftg, args.max_length)
                    if len(shortest_malignant_to_other) > 0:
                        print(f"Found {len(shortest_malignant_to_other)} pathways from {args.index_cell} to {cell}")

                        if args.index_cell in dict_Proteomics or cell in dict_Proteomics or cell in dict_ATACseq:
                            sender_protein = dict_Proteomics.get(args.index_cell, unified_gene_list)
                            receiver_protein = dict_Proteomics.get(cell, unified_gene_list)
                            receiver_tf = dict_ATACseq.get(cell, unified_gene_list)

                            result_multiomics = []
                            output_file = os.path.join(recon_dir1, f'{args.index_cell}_to_{cell}_pathway.txt')
                            with open(output_file, 'w') as f:
                                f.write(f"Ligand\tReceptor\tMediator\tTF\tTarget\n")
                                for pathway in shortest_malignant_to_other:
                                    col1 = pathway[0]  # Ligands
                                    col2 = pathway[1]  # Receptor
                                    col3 = pathway[2]  # Mediators
                                    col4 = pathway[3]  # TF
                                    col5 = pathway[4]  # TGs
                                    line = f"{col1}\t{col2}\t{col3}\t{col4}\t{col5}\n"
                                    if col1 in sender_protein and col2 in receiver_protein and col4 in receiver_tf:
                                        result_multiomics.append(line)
                                    f.write(line)

                            output_file = os.path.join(recon_dir2, f'{args.index_cell}_to_{cell}_pathway.txt')
                            with open(output_file, 'w') as f:
                                f.write(f"Ligand\tReceptor\tMediator\tTF\tTarget\n")
                                for line in result_multiomics:
                                    f.write(line)
                        else:
                            output_file = os.path.join(recon_dir1, f'{args.index_cell}_to_{cell}_pathway.txt')
                            with open(output_file, 'w') as f:
                                f.write(f"Ligand\tReceptor\tMediator\tTF\tTarget\n")
                                for pathway in shortest_malignant_to_other:
                                    col1 = pathway[0]  # Ligands
                                    col2 = pathway[1]  # Receptor
                                    col3 = pathway[2]  # Mediators
                                    col4 = pathway[3]  # TF
                                    col5 = pathway[4]  # TGs
                                    f.write(f"{col1}\t{col2}\t{col3}\t{col4}\t{col5}\n")

        print(f'{file} Signaling Pathway Inference Done')
