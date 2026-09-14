# SigFormer

## About:

SigFormer is a graph Transformer-based framework for reconstructing transcellular signaling pathways, including:

- Ligand–receptor interactions
- Six classes of intracellular signaling events
- Transcription factor (TF)–target gene regulation

SigFormer uses single-cell RNA sequencing (scRNA-seq) as the required input and can optionally integrate multi-omics and spatial data (if available) to refine pathway inference.

![](https://github.com/free1234hm/SigFormer/blob/main/Schematic.png)

## 1. Installation:

Clone the repository and enter the project directory:

```bash
git clone https://github.com/free1234hm/SigFormer.git
cd SigFormer
```

Extract `reference_library.zip` from the project root before running SigFormer:

```bash
unzip reference_library.zip
```

Create a Python virtual environment with Conda and install the required packages.

- `Numpy`: 1.26.4
- `Anndata`: 0.11.1
- `Scanpy`: 1.10.4
- `Torch (+CUDA118)`: torch 2.5.1 + cu118; torchvision 0.20.1 + cu118; torchaudio 2.5.1 + cu118
- `torch geometric`: 2.6.1
- `pyg-lib`: 0.4.0 + pt25cu118
- `torch-scatter`: 2.1.2 + pt25cu118
- `torch-sparse`: 0.6.18 + pt25cu118

**Example:**

```shell
conda create -n sigformerEnv python=3.12.8 pip
conda activate sigformerEnv
pip install numpy==1.26.4
pip install anndata==0.11.1
pip install scanpy==1.10.4
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```

or use the environment file:

```shell
pip install -r requirements.txt
```

On our local workstation (Intel Core i7-7800X CPU, 64 GB RAM, NVIDIA GeForce RTX 4090 24 GB), setting up the environment typically requires about 26 minutes, depending on internet speed and package source availability.

## 2. Prepare datasets

The GitHub test datasets are stored directly in `test_data/scRNA-seq/` (`.h5ad`), `test_data/scProteomics/` (`.txt`), and `test_data/scATAC-seq/` (`.tsv`).

### scRNA-seq data (required):

Provide scRNA-seq data as an .h5ad (AnnData) file.

- `adata.obs['celltype']` (required): used to infer bidirectional signaling pathways between any two cell types.
- `adata.obs['sample']` (optional): enables per-sample signaling inference. If multiple samples are present, SigFormer assumes the samples represent a similar biological state and integrates results into a cross-sample consensus. If `obs['sample']` is missing, SigFormer sets it to "merged" automatically.
- `adata.var`: gene metadata used for gene filtering and highly variable gene (HVG) identification.
- `adata.X`: expression matrix (sparse or dense).
- A numeric spatial-like entry in `adata.obsm` (optional), such as `adata.obsm['spatial']` or `adata.obsm['spatial_embeddings']`. Each row must correspond to one cell and follow the order of `adata.obs`. If no spatial embedding is available, set `--spatial` to `False`.

### Single-cell Proteomics input (optional):

Use `--scProteomics_path` to point to one tab-delimited file containing `celltype`, `protein`, and an optional `Score` column.

- Cell type names must exactly match the corresponding values in `obs['celltype']`.
- Multiple proteins with the same score can be separated by semicolons in the second column.
- If the entire file has no third column, all scores default to 1. If a Score column is present but a row is empty, that row receives a score of 0.
- Scores are normalized independently within each cell type.

### Single-cell ATAC-seq input (optional):

Use `--scATACseq_path` to point to one tab-delimited file containing `celltype`, `TF`, and an optional `Score` column.

- Cell type names must exactly match the corresponding values in `obs['celltype']`.
- Multiple TFs with the same score can be separated by semicolons in the second column.
- If the entire file has no third column, all scores default to 1. If a Score column is present but a row is empty, that row receives a score of 0.
- Scores are normalized independently within each cell type.

## 3. Run SigFormer

Below are the parameters used to run the provided test datasets (human cancers), via `SigFormer_main.py`.

### Required input files

- **scRNAseq_path** (`str`, default: `None`). Path to scRNA-seq data (an `.h5ad` file, or a folder containing `.h5ad` files).
- **pathway_file** (`str`, default: `./reference_library/Intracellular_signaling.txt`). Curated intracellular signaling interactions.
- **ligand_file** (`str`, default: `./reference_library/Ligand_secreted&membrane.txt`). Curated ligand–receptor pairs.

### Optional input files

- **scProteomics_path** (`str`, default: `None`). Path to a tab-delimited scProteomics file with an optional third-column score.
- **scATACseq_path** (`str`, default: `None`). Path to a tab-delimited scATAC-seq file with an optional third-column score.
- **retained_cell_types** (`str`, default: `None`). Optional TXT, TSV, or CSV file with one cell-type name per line or in the first column, matching `obs['celltype']` exactly (including spaces). The list must include `index_cell` and at least one other type. Omit it or pass an empty string to retain all annotated types, subject to quality and cell-count filters. Selection precedes per-sample preprocessing and HVG selection, so it can affect the selected HVGs.
- **background_gene_set** (`str`, default: `None`). Optional TXT, TSV, or CSV file whose first column contains custom background genes, such as genes of interest or user-derived differentially expressed genes. This set replaces HVG selection and is not capped by `hvg_top_gene`; genes must match `adata.var_names` and survive gene filtering. When omitted, the background is selected using `hvg_top_gene`.

### Data preprocessing

- **index_cell** (`str`, default: `Malignant`). Index cell type used as the reference for bidirectional pathway inference.
- **min_cell** (`float`, default: `0.01`). Gene filtering threshold: minimum fraction of cells a gene must be expressed in.
- **min_gene** (`float`, default: `0.01`). Cell filtering threshold: minimum fraction of genes a cell must express.
- **min_cell_count** (`int`, default: `5`). Minimum cell count for each cell type (cell types below this are filtered out).
- **normalize** (`bool`, default: `True`). Library-size normalization per cell.
- **log_trans** (`bool`, default: `True`). Log-transform expression.
- **hvg_top_gene** (`int`, default: `5000`). Number of highly variable genes (HVGs) used as background genes when `background_gene_set` is not provided.
- **cell_top_gene** (`int`, default: `500`). Number of top-expressed genes to keep per cell within the selected background gene set.

### Spatial mode

- **spatial** (`bool`, default: `False`). Enable spatial mode using a numeric spatial-like matrix in `adata.obsm`.
- **knn** (`int`, default: `10`). Number of nearest neighbors identified for each cell before self and same-cell-type neighbors are excluded.
- **membrane_ligand_file** (`str`, default: `./reference_library/Ligand_membrane.txt`). Membrane-bound ligand–receptor pairs used in spatial mode.
- **secreted_ligand_file** (`str`, default: `./reference_library/Ligand_secreted.txt`). Secreted ligand–receptor pairs used in spatial mode.
- **spatial_ligand_min_fraction** (`float`, default: `0.1`). A spatial ligand must be expressed in a fraction strictly greater than this value among the relevant neighboring or distant sender cells.

### Model training and network reconstruction

- **classification_accuracy** (`float`, default: `0.8`). Threshold for cell classification; samples below this threshold may be rejected.
- **ko_evidence_weight** (`float`, default: `1.0`). Weight assigned to the receptor-to-TF perturbation score in pathway evidence scores.
- **ligand_evidence_weight** (`float`, default: `1.0`). Weight assigned to sender ligand protein abundance.
- **receptor_evidence_weight** (`float`, default: `1.0`). Weight assigned to receiver receptor protein abundance.
- **tf_evidence_weight** (`float`, default: `1.0`). Weight assigned to receiver TF chromatin-binding potential.
- **edge_threshold** (`float`, default: `0.8`). Edge reconstruction threshold; keep edges with weights ≥ this value.
- **num_epochs** (`int`, default: `50`). Maximum number of training epochs; early stopping may terminate training sooner.
- **learning_rate** (`float`, default: `0.0001`). Learning rate for model optimization.
- **block_size** (`int`, default: `5000`). Target number of cell-type-stratified metacells used when a sample contains more cells than this value.
- **metacell_expr_threshold** (`float`, default: `0.05`). After metacell aggregation, mean expression values below this threshold are reset to zero; use `0` to disable thresholding.
- **random_seed** (`int`, default: `43`). Random seed for reproducibility.

**scRNA-seq inference example:**

```bash
python SigFormer_main.py --scRNAseq_path ./test_data/scRNA-seq/Data_Chung2017_Breast_all.h5ad --pathway_file ./reference_library/Intracellular_signaling.txt --ligand_file "./reference_library/Ligand_secreted&membrane.txt"
```

**Multi-omics integration example:**

```bash
python SigFormer_main.py --scRNAseq_path ./test_data/scRNA-seq/Data_Chung2017_Breast_all.h5ad --scProteomics_path ./test_data/scProteomics/Breast_cancer.txt --scATACseq_path ./test_data/scATAC-seq/Breast_cancer.tsv
```

**Example after creating the cell-type and background files:**

```bash
python SigFormer_main.py --scRNAseq_path ./test_data/scRNA-seq/Data_Chung2017_Breast_all.h5ad --retained_cell_types ./retained_cell_types.txt --background_gene_set ./background_genes.tsv
```

**Case-control analysis example:**

Supply a user-derived DEG list as the background. Analyze each condition separately; this option does not perform differential expression analysis.

```bash
python SigFormer_main.py --scRNAseq_path ./test_data/scRNA-seq/Data_Chung2017_Breast_all.h5ad --background_gene_set ./DEG_list.tsv
```

Repository directory and file names use underscores in place of whitespace. Gene symbols and cell-type labels inside data files are preserved. Quote user-supplied paths when they contain shell-special characters; the combined ligand-library filename contains `&`.

## 4. Check Results

SigFormer creates a `./result` folder in your current working directory containing inferred pathway files, including:

- `index_cell_to_cell_pathway.txt`: signaling pathways from the index cell type to a non-index cell type.
- `cell_to_index_cell_pathway.txt`: signaling pathways from a non-index cell type to the index cell type.
- ...

Reconstructed cell-type networks are written to `result/<dataset>/cell_network/`. Existing results from earlier versions are not renamed automatically.

Each pathway file is a tab-delimited text file, with one pathway per line and six columns: `Ligand, Receptor, Mediator, TF, Target, Evidence_score`. Rows are sorted by `Evidence_score` in descending order. In scRNA-seq-only results, `Evidence_score` equals the normalized receptor-to-TF knockout score. Raw knockout distances are first integrated across samples. After pathway inference, knockout scores are max-scaled across all pathways that will actually be written for the same receiver cell type; receptor self-effects, non-TF genes, and R-to-TF candidates that cannot form an inferred pathway do not contribute to the normalization maximum. In multi-omics results, `Evidence_score` is the available-modality weighted mean of the knockout score, sender ligand protein abundance, receiver receptor protein abundance, and receiver TF chromatin-binding potential. A missing modality is excluded from the denominator, whereas a feature not detected by an available modality receives a score of zero. Protein and chromatin-binding scores are max-scaled to `[0, 1]` within each cell type and input dataset. When several ligands share one inferred receptor-to-TF cascade and sender proteomics is available, every ligand with positive protein abundance is written as a separate row and scored with its own abundance. Undetected or zero-abundance ligands remain grouped in one semicolon-delimited row with a ligand score of zero.

`Mediator` encodes one or more shortest paths linking receptors to TFs:
- Multiple shortest paths of equal length are separated by semicolons `;`.
- Consecutive mediator genes within one path are separated by ASCII arrows `->`.
- Underscores remain part of gene names and are not treated as delimiters.

For example, `AKT1->MAPK1;PIK3CA->AKT1->MAPK1` represents two parallel receptor-to-TF cascades. An empty `Mediator` value indicates a direct receptor-to-TF edge without an intermediate signaling molecule.

## 5. Running Time

SigFormer is an end-to-end framework integrating cell-specific network construction, graph representation learning, network reconstruction, and in silico receptor knockout analysis. The runtimes below measure the complete workflow, rather than only model training or pathway inference. To improve computational efficiency, SigFormer uses sparse matrix operations where possible and GPU acceleration for graph representation learning. Large samples are compressed into cell-type-stratified metacells before network construction and model training, with a target size controlled by `--block_size` (default: 5,000).

We evaluated end-to-end runtime on 19 datasets containing 107–277,878 cells using the following hardware:

- **Local PC:** Intel Core i7-7800X CPU, 64 GB RAM, and an NVIDIA GeForce RTX 4090 GPU with 24 GB memory.
- **Server:** Two Intel Xeon Gold 6126 CPUs (24 physical cores and 48 threads in total), 128 GB RAM, and an NVIDIA Tesla V100 GPU with 16 GB memory.

| Hardware | 2,000 HVGs | 5,000 HVGs |
| --- | --- | --- |
| Local PC | 78–7,323 s (1.3 min–2.03 h) | 212–16,000 s (3.5 min–4.44 h) |
| Server | 70–2,953 s (1.2–49.2 min) | 164–6,125 s (2.7 min–1.70 h) |

Runtime increased with dataset size for small and moderately sized datasets. From 47,068 to 277,878 cells, however, runtime remained broadly stable, consistent with the effectiveness of the bounded metacell representation used for large samples. These benchmarks show that datasets containing hundreds of thousands of cells can be analyzed within a practical timeframe on the tested multi-core GPU server. Actual runtime depends on the input data, selected genes, analysis settings, and available computational resources; these measurements are not a guarantee for every dataset.

![End-to-end runtime of SigFormer](https://github.com/free1234hm/SigFormer/blob/main/Runtime.png)

Runtime of the complete SigFormer workflow across 19 datasets containing 107–277,878 cells using 2,000 or 5,000 highly variable genes, evaluated on (a) a local PC and (b) a server.

### Parameter guidance

- Use `--hvg_top_gene 2000` for a faster exploratory run, or the default `--hvg_top_gene 5000` for a broader background gene set. Reducing the gene set may decrease pathway coverage.
- `--cell_top_gene` (default: 500) controls the top-expressed genes selected per cell within the background. Reducing this value may reduce network-construction cost but also changes the inferred cell-specific networks.
- `--block_size` (default: 5,000) bounds the target metacell count for each large sample. Lower values can reduce downstream computation and memory requirements.
- When `--background_gene_set` is supplied, its retained genes replace HVG selection; changing `--hvg_top_gene` will not reduce that custom background.

We will continue to optimize computational efficiency while preserving predictive performance.

## Contact:

Han Mingfei: free1234hm@163.com

Zhu Yunping: zhuyunping@ncpsb.org.cn
