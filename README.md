# DHG-LGB: Disease-Hypergraph Integrated with LightGBM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17848044.svg)](https://doi.org/10.5281/zenodo.17848044)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

**Official implementation of "DHG-LGB: Predicting Metabolite-Disease Associations via Disease Hypergraph and LightGBM"**

## Overview

DHG-LGB is a novel computational framework for predicting metabolite-disease associations by integrating:
- **Hypergraph representation**: Diseases as hyperedges connecting metabolites, proteins, and Gene Ontology (GO) terms
- **Hypergraph Neural Networks (HGNN)**: Learning context-dependent embeddings through message passing
- **LightGBM classifier**: Efficient gradient boosting for binary classification

### Key Features

✅ **Higher-order interactions**: Captures multi-entity relationships beyond pairwise associations
✅ **Heterogeneous data integration**: Combines metabolites, proteins, and GO functional annotations
✅ **State-of-the-art performance**: AUC = 0.9983, MCC = 0.9305, AUPRC = 0.9860
✅ **Robust validation**: 5-fold cross-validation with consistent performance across varying data ratios

---

## Installation

### Prerequisites

- Python ≥ 3.8
- CUDA-compatible GPU (optional, for faster training)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/DHG-LGB.git
cd DHG-LGB

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Install from Source

```bash
pip install -e .
```

---

## Quick Start

### 1. Data Preparation

Download data from **HMDB 5.0** (https://hmdb.ca/) and **CTD** (http://ctdbase.org/):

```bash
# Place raw data in data/raw/
data/raw/
├── diseases.txt              # Disease mappings (MeSH IDs)
├── nodes.txt                 # Node mappings (metabolites, proteins, GO terms)
├── associations.txt          # Known metabolite-disease associations
├── metabolites_smiles.txt    # Metabolite chemical structures (SMILES)
├── proteins.fasta            # Protein amino acid sequences
└── go_ancestors.txt          # GO term hierarchical relationships
```

**Data Statistics** (from HMDB 5.0 and CTD 2023):
- Diseases: 178 (MeSH terms)
- Metabolites: 2,006 (HMDB IDs)
- Proteins: 4,912 (UniProt IDs)
- GO terms: 12,524
- Known associations: 4,000 metabolite-disease pairs

### 2. Preprocessing

Compute similarity matrices and generate negative samples:

```bash
python scripts/01_preprocess_data.py --config config/config.yaml
```

This step:
- Computes Tanimoto similarity for metabolites (based on Morgan fingerprints)
- Computes BLAST sequence similarity for proteins
- Computes GO semantic similarity using ancestral contribution method
- Generates negative samples with indirect association filtering

### 3. Train Hypergraph Neural Network

Learn node and hyperedge embeddings:

```bash
python scripts/02_train_hgnn.py --config config/config.yaml
```

Output:
- `data/embeddings/node_embeddings.txt` (19,442 nodes × 500 dimensions)
- `data/embeddings/disease_embeddings.txt` (178 diseases × 500 dimensions)

### 4. Train LightGBM Classifier

Train the prediction model with 5-fold cross-validation:

```bash
python scripts/03_train_classifier.py --config config/config.yaml
```

### 5. Evaluate Performance

Compute comprehensive metrics:

```bash
python scripts/04_evaluate_model.py --config config/config.yaml
```

Metrics computed:
- AUC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- MCC (Matthews Correlation Coefficient)
- Accuracy, Sensitivity, Specificity, Precision

---

## Project Structure

```
DHG-LGB/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Installation script
├── CITATION.bib                       # Citation information
├── config/
│   └── config.yaml                    # Configuration parameters
├── data/
│   ├── raw/                           # Raw data files
│   ├── processed/                     # Processed data
│   └── embeddings/                    # Learned embeddings
├── src/
│   ├── preprocessing/                 # Data preprocessing modules
│   │   ├── similarity.py              # Similarity matrix computation
│   │   ├── negative_sampling.py       # Negative sample generation
│   │   └── feature_extraction.py      # Feature engineering
│   ├── models/
│   │   ├── hgnn.py                    # Hypergraph Neural Network
│   │   └── classifier.py              # LightGBM classifier
│   ├── training/
│   │   ├── train_hgnn.py              # HGNN training script
│   │   └── train_classifier.py        # Classifier training script
│   ├── evaluation/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── cross_validation.py        # Cross-validation utilities
│   ├── visualization/
│   │   └── hypergraph_viz.py          # Hypergraph visualization
│   └── utils/
│       ├── logger.py                  # Logging utilities
│       ├── io.py                      # File I/O functions
│       └── config_loader.py           # Configuration loader
├── scripts/                           # Executable scripts
│   ├── 01_preprocess_data.py
│   ├── 02_train_hgnn.py
│   ├── 03_train_classifier.py
│   └── 04_evaluate_model.py
└── results/                           # Output results
    ├── figures/                       # Visualization plots
    ├── metrics/                       # Performance metrics
    └── predictions/                   # Prediction results
```

---

## Configuration

Modify `config/config.yaml` to customize:

```yaml
# Example configuration
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  embeddings_dir: "data/embeddings"

hgnn:
  embedding_dim: 500
  num_layers: 2
  dropout: 0.4
  weight_decay: 5e-5
  learning_rate: 0.001
  epochs: 200
  early_stopping_patience: 10

classifier:
  algorithm: "lightgbm"
  n_estimators: 100
  max_depth: -1
  learning_rate: 0.1
  reg_lambda: 0.1  # L2 regularization
  num_leaves: 31

cross_validation:
  n_folds: 5
  random_state: 42
  shuffle: true
```

---

## Results

### Performance Metrics (5-fold Cross-Validation)

| Metric | DHG-LGB | XGBoost | GBDT | MLP | AdaBoost | Random Forest |
|--------|---------|---------|------|-----|----------|---------------|
| **MCC** | **0.9305 ± 0.0012** | 0.9270 ± 0.0016 | 0.9195 ± 0.0023 | 0.9095 ± 0.0032 | 0.8945 ± 0.0038 | 0.8995 ± 0.0029 |
| **AUC** | **0.9983 ± 0.0001** | 0.9976 ± 0.0002 | 0.9961 ± 0.0003 | 0.9867 ± 0.0006 | 0.9812 ± 0.0008 | 0.9823 ± 0.0007 |
| **AUPRC** | **0.9860 ± 0.0003** | 0.9823 ± 0.0005 | 0.9765 ± 0.0008 | 0.9624 ± 0.0012 | 0.9414 ± 0.0015 | 0.9556 ± 0.0011 |
| Accuracy | 98.87 ± 0.05 | 98.71 ± 0.07 | 98.42 ± 0.11 | 97.95 ± 0.18 | 97.23 ± 0.22 | 97.58 ± 0.16 |

### Comparison with Baseline Methods

| Method | AUC | AUPRC |
|--------|-----|-------|
| **DHG-LGB** | **0.9978** | **0.9808** |
| PageRank | - | - |
| KATZ | - | - |
| EKRR | - | 0.398 |
| GCNAT | - | - |
| MDA-AENMF | - | - |

*Note: Baseline method metrics extracted from original publications due to reproduction challenges.*

### Case Study Validation

| Disease | Validated Predictions | Validation Rate |
|---------|----------------------|-----------------|
| Obesity | 9/10 | 90% |
| Schizophrenia | 10/10 | 100% |
| Crohn's Disease | 10/10 | 100% |
| **Overall** | **29/30** | **96.7%** |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024dhglgb,
  title={DHG-LGB: Predicting Metabolite-Disease Associations via Disease Hypergraph and LightGBM},
  author={Your Name and Coauthors},
  journal={Journal Name},
  year={2024},
  doi={10.xxxx/xxxxx}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please contact:
- **Your Name**: your.email@institution.edu
- **Lab Website**: https://yourlab.institution.edu

---

## Acknowledgments

- Data sources: [HMDB 5.0](https://hmdb.ca/) and [CTD](http://ctdbase.org/)
- Hypergraph library: [DeepHypergraph (DHG)](https://github.com/iMoonLab/DeepHypergraph)
- Gradient boosting: [LightGBM](https://github.com/microsoft/LightGBM)

---

## Changelog

### Version 1.0.0 (2024-12-07)
- Initial public release
- Complete implementation of DHG-LGB framework
- 5-fold cross-validation with comprehensive metrics
- Case study validation for 3 representative diseases
