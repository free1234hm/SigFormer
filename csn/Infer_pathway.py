from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np


def find_shortest_path(G, source, target, max_length):
    """Find a path between source and target nodes using NetworkX."""
    try:
        shortest_paths = list(nx.all_shortest_paths(G, source=source, target=target))
        if len(shortest_paths) > 0 and len(shortest_paths[0]) <= max_length:
            return shortest_paths
        else:
            return None
    except nx.NetworkXNoPath:
        return None


def find_diverse_paths(G, source, target, max_length):
    result_paths = []
    G_copy = G.copy()
    while True:
        try:
            paths = list(nx.all_shortest_paths(G_copy, source, target))
            path_length = len(paths[0]) - 1
            if path_length > max_length:
                break
            result_paths.extend(paths)
            # Remove edges of the found paths
            for path in paths:
                edges = list(zip(path[:-1], path[1:]))
                G_copy.remove_edges_from(edges)
        except nx.NetworkXNoPath:
            break
    return result_paths


def merge_shortest_paths(unified_gene_list, shortest_path, tftg_malignant):
    if not shortest_path:
        return None  # 没有任何路径

    # 如果只有一个 path，则直接返回
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

    # 所有 paths 的 source 和 TF
    sources = [p[0] for p in shortest_path]
    tfs = [p[-1] for p in shortest_path]

    # 检查是否全部 source 相同 & 全部 TF 相同
    if len(set(sources)) > 1 or len(set(tfs)) > 1:
        raise ValueError("shortest_path 中包含不同的 source 或 TF，无法 merge")

    source = sources[0]
    tf = tfs[0]
    tgs = tftg_malignant.col[tftg_malignant.row == tf].tolist()
    tg_names = ";".join([unified_gene_list[i] for i in tgs])

    # 合并中间部分
    mid_strings = []
    for p in shortest_path:
        sub_p = [unified_gene_list[i] for i in p]
        middle = sub_p[1:-1]  # 去掉头尾
        # 如果中间只有一个元素也可以 join
        mid_str = "_".join(middle) if middle else ""
        mid_strings.append(mid_str)

    # 用 ";" 连起来
    merged_middle_str = ";".join(mid_strings)

    merged_path = [unified_gene_list[source], merged_middle_str, unified_gene_list[tf], tg_names]
    return merged_path


def infer_pathway(unified_gene_list, gene_set, lg_rp_dict, A3_sparse, A4_sparse, max_length=10):
    G = nx.from_numpy_array(A3_sparse.toarray(), create_using=nx.DiGraph)
    lg_rp_pairs = [(k, v) for k, s in lg_rp_dict.items() for v in s]

    pathways_lg_rp = [
        [ligand, receptor]
        for ligand in gene_set
        for _, receptor in lg_rp_pairs if ligand == _
    ]

    source_set = set(p[-1] for p in pathways_lg_rp)  # set of receptors
    target_set = set(A4_sparse.row)  # Genes with outgoing edges in A4

    dict_rplg = {}
    for ligand, receptor in pathways_lg_rp:
        dict_rplg.setdefault(receptor, []).append(ligand)

    shortest_pathways = []
    for source in source_set:
        ligands_for_source = dict_rplg.get(source, [])
        ligands_for_source = [unified_gene_list[i] for i in ligands_for_source]
        merged_ligs = ";".join(ligands_for_source)
        for target in target_set:
            if source != target:
                shortest_path = find_shortest_path(G, source=source, target=target, max_length=max_length)
                if shortest_path:
                    merged = merge_shortest_paths(unified_gene_list, shortest_path, A4_sparse)
                    merged.insert(0, merged_ligs)
                    shortest_pathways.append(merged)
    return shortest_pathways

