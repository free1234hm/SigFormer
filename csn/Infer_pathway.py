# File path: SigFormer_v3/csn/Infer_pathway.py
from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np


PATH_NODE_SEPARATOR = "->"
PARALLEL_PATH_SEPARATOR = ";"


def subgraph_fast(A3_sparse, perturbed_gene):
    """
    Keep a coo_matrix with the original shape and retain only edges whose
    row and column are both in perturbed_gene.
    """
    keep = np.zeros(A3_sparse.shape[0], dtype=bool)
    keep[list(perturbed_gene)] = True

    mask = keep[A3_sparse.row] & keep[A3_sparse.col]

    return coo_matrix(
        (A3_sparse.data[mask], (A3_sparse.row[mask], A3_sparse.col[mask])),
        shape=A3_sparse.shape
    )


def coo_to_digraph(sub_sparse):
    G = nx.DiGraph()
    if sub_sparse.nnz > 0:
        safe_weights = np.clip(sub_sparse.data, 1e-12, 1.0)
        costs = 0.5 - np.log(safe_weights)

        edges = zip(sub_sparse.row.tolist(), sub_sparse.col.tolist(), costs.tolist())
        G.add_weighted_edges_from(edges, weight='cost')
    return G


def build_all_shortest_paths_from_pred(pred, source, target, max_paths=100):
    """
    Backtrack all minimum-cost parallel paths from source to target using
    the predecessor dictionary returned by nx.dijkstra_predecessor_and_distance.
    max_paths prevents path explosion.
    """
    if target not in pred:
        return None

    if target == source:
        return [[source]]

    path_count = [0]

    def backtrack(node):
        if path_count[0] >= max_paths:
            return []

        if node == source:
            path_count[0] += 1
            return [[source]]

        paths = []
        for p in pred.get(node, []):
            for path in backtrack(p):
                paths.append(path + [node])
                if path_count[0] >= max_paths:
                    break
            if path_count[0] >= max_paths:
                break
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
        mid_str = PATH_NODE_SEPARATOR.join(middle) if middle else ""
        tgs = tftg_malignant.col[tftg_malignant.row == tf].tolist()
        tg_names = ";".join([unified_gene_list[i] for i in tgs]) if tgs else "No_Target"
        merged_path = [unified_gene_list[source], mid_str, unified_gene_list[tf], tg_names]
        return merged_path

    sources = [p[0] for p in shortest_path]
    tfs = [p[-1] for p in shortest_path]

    if len(set(sources)) > 1 or len(set(tfs)) > 1:
        raise ValueError("shortest_path contains different sources or TFs and cannot be merged")

    source = sources[0]
    tf = tfs[0]
    tgs = tftg_malignant.col[tftg_malignant.row == tf].tolist()
    tg_names = ";".join([unified_gene_list[i] for i in tgs]) if tgs else "No_Target"

    mid_strings = []
    for p in shortest_path:
        sub_p = [unified_gene_list[i] for i in p]
        middle = sub_p[1:-1]
        mid_str = PATH_NODE_SEPARATOR.join(middle) if middle else ""
        mid_strings.append(mid_str)

    merged_middle_str = PARALLEL_PATH_SEPARATOR.join(mid_strings)
    merged_path = [unified_gene_list[source], merged_middle_str, unified_gene_list[tf], tg_names]
    return merged_path


def infer_pathway(
    unified_gene_list,
    gene_set,
    perturbed_gene,
    lg_rp_dict,
    A3_sparse,
    A4_sparse
):
    target_set = set(A4_sparse.row)  # TFs

    dict_rplg = {}
    for ligand in sorted(set(gene_set)):
        for receptor in sorted(lg_rp_dict.get(ligand, ())):
            dict_rplg.setdefault(receptor, []).append(ligand)

    pathways_with_perturbed_TFs = []
    for source in sorted(dict_rplg):
        ligands_for_source = dict_rplg.get(source, [])
        ligands_for_source = [unified_gene_list[i] for i in ligands_for_source]
        merged_ligs = ";".join(ligands_for_source)

        if source not in perturbed_gene:
            continue

        receptor_perturbed_gene = set(perturbed_gene[source])
        receptor_perturbed_gene.add(source)

        sub_sparse = subgraph_fast(A3_sparse, receptor_perturbed_gene)
        if sub_sparse.nnz == 0:
            continue

        G = coo_to_digraph(sub_sparse)
        if source not in G:
            continue

        # Use the weighted predecessor algorithm to collect all optimal predecessors.
        try:
            pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight='cost')
        except nx.NetworkXNoPath:
            continue

        candidate_targets = target_set & receptor_perturbed_gene
        if source in candidate_targets:
            candidate_targets.remove(source)

        for target in sorted(candidate_targets):
            if target not in pred:
                continue

            # Backtrack all optimal paths.
            shortest_paths = build_all_shortest_paths_from_pred(pred, source, target, max_paths=100)

            if not shortest_paths:
                continue
            shortest_paths.sort(key=tuple)

            merged = merge_shortest_paths(unified_gene_list, shortest_paths, A4_sparse)
            if merged:
                target_scores = perturbed_gene[source]
                if isinstance(target_scores, dict):
                    perturbation_score = float(target_scores[target])
                else:
                    # Backward compatibility for legacy significant-gene lists.
                    perturbation_score = 1.0
                merged.insert(0, merged_ligs)
                merged.append(perturbation_score)
                pathways_with_perturbed_TFs.append(merged)

    return sorted(pathways_with_perturbed_TFs, key=lambda pathway: pathway[-1], reverse=True)
