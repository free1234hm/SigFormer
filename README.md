# SigFormer

## About:

SigFormer is a graph Transformer-based framework for reconstructing transcellular signaling pathways, including:

- Ligand–receptor interactions
- Six classes of intracellular signaling events
- Transcription factor (TF)–target gene regulation

SigFormer uses scRNA-seq as the required input and can optionally integrate multi-omics and spatial data (if available) to refine pathway inference.

## 1. Installation:

Clone the repository and enter the project directory:

```bash
git clone https://github.com/free1234hm/SigFormer.git
cd SigFormer
```

Create a Python virtual environment with Conda and install the required packages.

- `Numpy`: 1.26.4
- `Scanpy`: 1.10.4 or above
- `Anndata`: 0.11.1 or above
- `Torch (+CUDA118)`: torch 2.5.1 + cu118; torchvision 0.20.1 + cu118; torchaudio 2.5.1 + cu118
- `torch geometric`: 2.6.1
- `pyg-lib`: 0.4.0 + pt25cu118
- `torch-scatter`: 2.1.2 + pt25cu118
- `torch-sparse`: 0.6.18 + pt25cu118

**Example:**

```shell
conda create -n sigformerEnv python=3.12 pip
conda activate sigformerEnv
pip install -r requirements.txt
```

## 2. Prepare datasets

### scRNA-seq data (required):

Provide scRNA-seq data as an .h5ad (AnnData) file.

- `adata.obs['celltype']` (required): used to infer bidirectional signaling pathways between any two cell types.
- `adata.obs['sample']` (optional): enables per-sample signaling inference. If multiple samples are present, SigFormer assumes the samples represent a similar biological state and integrates results into a cross-sample consensus. If `obs['sample']` is missing, SigFormer sets it to "merged" automatically.
- `adata.var`: gene metadata used for gene filtering and highly variable gene (HVG) identification.
- `adata.X`: expression matrix (sparse or dense).
- `adata.obsm['spatial']` (optional): spatial embeddings. Each row corresponds to one cell and must match the order of cells in adata.obs. If your data does not include spatial embeddings, set --spatial to False.

### Single-cell Proteomics input (optional):

Use `--scProteomics_path` to point to a directory containing single-cell proteomics files in `.txt` format, where each file stores the protein expression profile for one cell type.

- Each filename must exactly match the corresponding cell type name in `obs['celltype']`. Example: if `obs['celltype']` = "B_cell", the file must be `B_cell.txt`.
- For example directory structure and file format, refer to: `test data/scProteomics`.

### Single-cell ATAC-seq input (optional):

Use `--scATACseq_path` to point to a directory containing single-cell ATAC-seq files, where each file stores TFs activated in a specific cell type.

- For examples of file format and directory structure, refer to: `test data/scATAC-seq`.

## 3. Run SigFormer

Below are the parameters used to run the provided test datasets (human cancers), via `SigFormer_main.py`.

### Required input files

- **scRNAseq_path** (`str`, default: `None`). Path to scRNA-seq data (an .h5ad file, or a folder containing .h5ad files).
- **pathway_file** (`str`, default: `./reference library/Intracellular signaling.txt`). Curated intracellular signaling interactions.
- **ligand_file** (`str`, default: `./reference library/Ligand_secreted&membrane.txt`). Curated ligand–receptor pairs.

### Optional input files

- **scProteomics_path** (`str`, default: `None`). Path to scProteomics data (a folder containing cell-type-specific .txt files).
- **scATACseq_path** (`str`, default: `None`). Path to scATAC-seq data (a folder containing cell-type-specific .txt files).

### Data preprocessing

- **index_cell** (`str`, default: `Malignant`). Index cell type used as the reference for bidirectional pathway inference.
- **min_cell** (`float`, default: `0.01`). Gene filtering threshold: minimum fraction of cells a gene must be expressed in.
- **min_gene** (`float`, default: `0.01`). Cell filtering threshold: minimum fraction of genes a cell must express.
- **min_cell_count** (`int`, default: `5`). Minimum cell count for each cell type (cell types below this are filtered out).
- **normalize** (`bool`, default: `True`). Library-size normalization per cell.
- **log_trans** (`bool`, default: `True`). Log-transform expression (log1p).
- **hvg_top_gene** (`int`, default: `5000`). Number of highly variable genes (HVGs) to keep.
- **cell_top_gene** (`int`, default: `500`). Number of top expressed genes to keep per cell.

### Spatial mode

- **spatial** (`bool`, default: `False`). Enable spatial mode (requires adata.obsm['spatial']).
- **knn** (`int`, default: `10`). If spatial mode is enabled, use the k nearest neighbors for each index cell.

### Model training and network reconstruction

- **classification_accuracy** (`float`, default: `0.8`). Threshold for cell classification; samples below this threshold may be rejected.
- **edge_threshold** (`float`, default: `0.8`). Edge reconstruction threshold; keep edges with weights ≥ this value.
- **num_epochs** (`int`, default: `20`). Training epochs (increase if accuracy improves slowly).
- **learning_rate** (`float`, default: `0.001`). Learning rate (decrease if accuracy fluctuates).
- **block_size** (`int`, default: `5000`). Block size for large datasets (splits into blocks to reduce memory usage).
- **random_seed** (`int`, default: `43`). Random seed for reproducibility.
- **max_length** (`int`, default: `5`). Maximum number of receptor-to-TF steps in inferred signaling pathways.

**scRNA-seq inference example:** :

```bash
python SigFormer_main.py --scRNAseq_path "./test data/scRNA-seq/Data_Chung2017_Breast_all.h5ad"
```

**Multi-omics integration example:** :

```bash
python SigFormer_main.py --scRNAseq_path "./test data/scRNA-seq/Data_Chung2017_Breast_all.h5ad" --scProteomics_path "./test data/scProteomics/Breast" --scATACseq_path "./test data/scATAC-seq/Breast"
```

## 4. Check Results

SigFormer creates a `./result` folder in your current working directory containing inferred pathway files, including:

- `index_cell_to_cell_pathway.txt`: signaling pathways from the index cell type to a non-index cell type.
- `cell_to_index_cell_pathway.txt`: signaling pathways from a non-index cell type to the index cell type.
- ...

Each pathway file is a tab-delimited text file, with one pathway per line and five columns: `Ligand, Receptor, Mediator, TF, Target`.

`Mediator` encodes one or more shortest paths linking receptors to TFs:
- Multiple shortest paths of equal length are separated by semicolons `;`.
- Nodes within a single shortest path are separated by underscores `_`.

## Contact:

Han Mingfei: free1234hm@163.com

Zhu Yunping: zhuyunping@ncpsb.org.cn
