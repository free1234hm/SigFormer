from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np


def subgraph_fast(A3_sparse, perturbed_gene):
    """
    保持 coo_matrix，形状不变，只保留 row/col 都在 perturbed_gene 中的边
    """
    keep = np.zeros(A3_sparse.shape[0], dtype=bool)
    keep[list(perturbed_gene)] = True

    mask = keep[A3_sparse.row] & keep[A3_sparse.col]

    return coo_matrix(
        (A3_sparse.data[mask], (A3_sparse.row[mask], A3_sparse.col[mask])),
        shape=A3_sparse.shape
    )


def coo_to_digraph(sub_sparse):
    """
    直接从 COO 边表构建 DiGraph，不转 dense
    """
    G = nx.DiGraph()
    if sub_sparse.nnz > 0:
        G.add_edges_from(zip(sub_sparse.row.tolist(), sub_sparse.col.tolist()))
    return G


def build_all_shortest_paths_from_pred(pred, source, target):
    """
    根据 nx.predecessor 返回的前驱字典，回溯 source->target 的所有 shortest paths
    pred[v] = [u1, u2, ...] 表示 shortest path 上 v 的前驱
    """
    if target not in pred:
        return None

    # source 的 predecessor 通常是 []
    if target == source:
        return [[source]]

    def backtrack(node):
        if node == source:
            return [[source]]
        paths = []
        for p in pred[node]:
            for path in backtrack(p):
                paths.append(path + [node])
        return paths

    return backtrack(target)


def merge_shortest_paths(unified_gene_list, shortest_path, tftg_malignant):
    if not shortest_path:
        return None

    if len(shortest_path) == 1:
        p = shortest_path[0]
        source = p[0]
        tf = p[-1]
        middle = [unified_gene_list[i] for i in p[1:-1]]
        mid_str = "_".join(middle) if middle else ""
        tgs = tftg_malignant.col[tftg_malignant.row == tf].tolist()
        tg_names = ";".join([unified_gene_list[i] for i in tgs])
        merged_path = [unified_gene_list[source], mid_str, unified_gene_list[tf], tg_names]
        return merged_path

    sources = [p[0] for p in shortest_path]
    tfs = [p[-1] for p in shortest_path]

    if len(set(sources)) > 1 or len(set(tfs)) > 1:
        raise ValueError("shortest_path 中包含不同的 source 或 TF，无法 merge")

    source = sources[0]
    tf = tfs[0]
    tgs = tftg_malignant.col[tftg_malignant.row == tf].tolist()
    tg_names = ";".join([unified_gene_list[i] for i in tgs])

    mid_strings = []
    for p in shortest_path:
        sub_p = [unified_gene_list[i] for i in p]
        middle = sub_p[1:-1]
        mid_str = "_".join(middle) if middle else ""
        mid_strings.append(mid_str)

    merged_middle_str = ";".join(mid_strings)
    merged_path = [unified_gene_list[source], merged_middle_str, unified_gene_list[tf], tg_names]
    return merged_path


def infer_pathway(unified_gene_list, gene_set, perturbed_gene, lg_rp_dict, A3_sparse, A4_sparse, max_length=10):
    lg_rp_pairs = [(k, v) for k, s in lg_rp_dict.items() for v in s]
    pathways_lg_rp = [
        [ligand, receptor]
        for ligand in gene_set
        for _, receptor in lg_rp_pairs if ligand == _
    ]

    source_set = set(p[-1] for p in pathways_lg_rp)   # receptors
    target_set = set(A4_sparse.row)                   # TFs

    dict_rplg = {}
    for ligand, receptor in pathways_lg_rp:
        dict_rplg.setdefault(receptor, []).append(ligand)

    pathways_with_perturbed_TFs = []

    for source in source_set:
        ligands_for_source = dict_rplg.get(source, [])
        ligands_for_source = [unified_gene_list[i] for i in ligands_for_source]
        merged_ligs = ";".join(ligands_for_source)

        # 如果这个 receptor 没有 perturbation 结果，跳过
        if source not in perturbed_gene:
            continue

        receptor_perturbed_gene = set(perturbed_gene[source])
        receptor_perturbed_gene.add(source)  # add receptor itself

        # 只构建一次该 receptor 的子图
        sub_sparse = subgraph_fast(A3_sparse, receptor_perturbed_gene)
        if sub_sparse.nnz == 0:
            continue

        # 直接由 COO 边表建图，不 toarray()
        G = coo_to_digraph(sub_sparse)
        if source not in G:
            continue

        # 一次性得到 source 到所有节点的最短路前驱信息
        pred = nx.predecessor(G, source, cutoff=max_length - 1)

        # 只保留当前子图中存在的 target
        candidate_targets = target_set & receptor_perturbed_gene
        if source in candidate_targets:
            candidate_targets.remove(source)

        for target in candidate_targets:
            if target not in pred:
                continue

            shortest_path = build_all_shortest_paths_from_pred(pred, source, target)
            if shortest_path:
                merged = merge_shortest_paths(unified_gene_list, shortest_path, A4_sparse)
                merged.insert(0, merged_ligs)
                pathways_with_perturbed_TFs.append(merged)

    return pathways_with_perturbed_TFs