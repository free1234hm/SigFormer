import pandas as pd
import scanpy as sc
from pathlib import Path

cancer_list = ['Breast cancer', 'Colorectal cancer', 'Glioblastoma',
               'Leukemia', 'Lung cancer', 'Melanoma',
               'Ovarian cancer', 'Pancreatic cancer']

for cancertype in cancer_list:
    background_genes = set()
    adata_file = f"./Evaluation_data/{cancertype}.h5ad"
    adata = sc.read(adata_file)
    background_genes = adata.var_names.to_list()
    # print("Background genes: " + str(len(background_genes)))

    tf_file = r'./Background_files/TF_names_v_1.01.txt'
    background_tfs = set()
    with open(tf_file, "r", encoding="utf-8") as f:
        for line in f:
            tf = line.strip().replace('"', '')
            if tf in background_genes:
                background_tfs.add(tf)
    # print("Background TFs", str(len(background_tfs)))

    ligand_file = r'./Background_files/Ligand_secrete&membrane.txt'
    background_ligands = set()
    with open(ligand_file, "r", encoding="utf-8") as f:
        for line in f:
            ll = line.strip().replace('"', '')
            if ll in background_genes:
                background_ligands.add(ll)
    # print("Background ligands", str(len(background_ligands)))

    pair_file = f"./Positive_data/ligand-receptor/{cancertype}.txt"
    cancer_ligand_genes = set()
    cancer_receptor_genes = set()
    cancer_lgrp_pairs = set()
    with open(pair_file, "r", encoding="utf-8") as f:
        for line in f:
            ll = line.strip().split("\t")
            if len(ll) > 1:
                gene1, gene2 = ll[0], ll[1]
                if gene1 in background_ligands and gene2 in background_genes:
                    cancer_ligand_genes.add(gene1)
                    cancer_receptor_genes.add(gene2)
                    cancer_lgrp_pairs.add(f'{gene1}_{gene2}')
            else:
                print(line)
    # print("Positive_ligand_genes: " + str(len(cancer_ligand_genes)))
    # print("Positive_receptor_genes: " + str(len(cancer_receptor_genes)))

    pair_file = f"./Positive_data/receptor-TF/{cancertype}.txt"
    cancer_mediator_genes1 = set()
    with open(pair_file, "r", encoding="utf-8") as f:
        for line in f:
            ll = line.strip().split("\t")
            if len(ll) > 2:
                gene1, gene2, ttype = ll[0], ll[1], ll[2]
                if gene1 in background_genes and gene2 in background_genes:
                    cancer_mediator_genes1.add(gene1)
                    cancer_mediator_genes1.add(gene2)
            else:
                print(line)
    # print("Positive_mediator_genes: " + str(len(cancer_mediator_genes1)))

    pair_file = f"./Positive_data/TF-target/{cancertype}.txt"
    cancer_tf_genes = set()
    cancer_genes = set()
    with open(pair_file, "r", encoding="utf-8") as f:
        for line in f:
            ll = line.strip().split("\t")
            if len(ll) > 1:
                gene1, gene2 = ll[0], ll[1]
                if gene1 in background_tfs and gene2 in background_genes:
                    cancer_tf_genes.add(gene1)
                    cancer_genes.add(gene2)
            else:
                print(line)
    # print("Positive_tf_genes: " + str(len(cancer_tf_genes)))
    # print("Positive_tg_genes: " + str(len(cancer_genes)))

    deeppath_file1 = f"./Pathway_file/{cancertype}"
    deeppath_file1 = Path(deeppath_file1)

    all_ligand_genes = set()
    all_receptor_genes = set()
    all_mediator_genes = set()
    all_tf_genes = set()
    all_tg_genes = set()

    for file in deeppath_file1.rglob('*'):
        if file.is_file() and '_to_' in file.name:
            with open(file, "r") as f:
                line = f.readline().strip()
                for line in f:
                    ll = line.strip().split("\t")
                    if len(ll) > 1:
                        ligand, receptor, mediators, tf, tgs = ll[0], ll[1], ll[2], ll[3], ll[4]
                        for lg in ligand.split(';'):
                            if lg in background_ligands:
                                all_ligand_genes.add(lg)

                        all_receptor_genes.add(receptor)
                        all_mediator_genes.add(receptor)
                        all_mediator_genes.add(tf)
                        if len(mediators) > 0:
                            for m1 in mediators.split(';'):
                                for m2 in m1.split('_'):
                                    all_mediator_genes.add(m2)

                        if tf in background_tfs and len(tgs) > 0:
                            all_tf_genes.add(tf)
                            for tg in tgs.split(';'):
                                all_tg_genes.add(tg)

    print(f'{cancertype}')
    print('Stage' + '\t' + 'Recall' + '\t' + 'Precision' + '\t' + 'F1')

    TP = all_ligand_genes & cancer_ligand_genes
    recall1 = len(TP) / len(cancer_ligand_genes)
    precision1 = len(TP) / len(all_ligand_genes)
    if len(TP) > 0:
        F1 = 2 * (recall1 * precision1) / (recall1 + precision1)
    else:
        F1 = 0
    print('Ligand' + '\t' + str(recall1) + '\t' + str(precision1) + '\t' + str(F1))

    TP = all_receptor_genes & cancer_receptor_genes
    recall1 = len(TP) / len(cancer_receptor_genes)
    precision1 = len(TP) / len(all_receptor_genes)
    if len(TP) > 0:
        F1 = 2 * (recall1 * precision1) / (recall1 + precision1)
    else:
        F1 = 0
    print('Receptor' + '\t' + str(recall1) + '\t' + str(precision1) + '\t' + str(F1))

    TP = all_mediator_genes & cancer_mediator_genes1
    if len(TP) > 0:
        recall1 = len(TP) / len(cancer_mediator_genes1)
        precision1 = len(TP) / len(all_mediator_genes)
        F1 = 2 * (recall1 * precision1) / (recall1 + precision1)
    else:
        recall1 = 0
        precision1 = 0
        F1 = 0
    print('Mediator' + '\t' + str(recall1) + '\t' + str(precision1) + '\t' + str(F1))

    TP = all_tf_genes & cancer_tf_genes
    recall1 = len(TP) / len(cancer_tf_genes)
    precision1 = len(TP) / len(all_tf_genes)
    if len(TP) > 0:
        F1 = 2 * (recall1 * precision1) / (recall1 + precision1)
    else:
        F1 = 0
    print('TF:' + '\t' + str(recall1) + '\t' + str(precision1) + '\t' + str(F1))

    TP = all_tg_genes & cancer_genes
    recall1 = len(TP) / len(cancer_genes)
    precision1 = len(TP) / len(all_tg_genes)
    if len(TP) > 0:
        F1 = 2 * (recall1 * precision1) / (recall1 + precision1)
    else:
        F1 = 0
    print('Target:' + '\t' + str(recall1) + '\t' + str(precision1) + '\t' + str(F1))

