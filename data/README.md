# Data Directory - Complete DHG-LGB Dataset

This directory contains all data files used in the DHG-LGB study, organized into raw and processed subdirectories.

**Total Size:** ~445 MB
**Manuscript:** "Identifying Metabolite-Disease Associations via Messaging in Hypergraphs"

---

## Directory Structure

```
data/
├── raw/                                (16 MB)
│   ├── HMDB_data_raw.txt              5.7 MB - HMDB metabolite-disease associations
│   ├── CTD_data_raw.txt               4.7 MB - CTD disease-metabolite associations
│   ├── protein_sequences.fasta        3.0 MB - Protein amino acid sequences
│   ├── GO_ancestors_DAG.txt           2.6 MB - Gene Ontology hierarchy
│   ├── disease_to_entity_mapping.txt  2.4 MB - Disease-entity connections
│   ├── HMDB_SMILES_structures.txt     152 KB - Chemical structures
│   └── README.md                            - Detailed raw data documentation
│
├── processed/
│   ├── similarity_matrices/           (235 MB)
│   │   ├── metabolite_similarity_2006x2006.txt    25 MB
│   │   ├── protein_similarity_4912x4912.txt       60 MB
│   │   ├── GO_similarity_12524x12524.txt         150 MB
│   │   └── README.md                              - Similarity computation details
│   │
│   ├── hypergraph_structure/          (10 MB)
│   │   ├── hypergraph_incidence_matrix_178x19442.txt  6.7 MB
│   │   ├── associations_with_entity_names.txt         3.7 MB
│   │   ├── node_index_to_name_mapping.txt             479 KB
│   │   ├── positive_samples_4000.txt                   39 KB
│   │   ├── disease_hyperedge_indices.txt              3.1 KB
│   │   └── README.md                                   - Hypergraph construction details
│   │
│   └── embeddings/                    (180 MB)
│       ├── HGNN_node_embeddings_19620x500.txt    178 MB
│       ├── HGNN_disease_embeddings_178x500.txt     1.7 MB
│       └── README.md                               - HGNN training and usage details
│
└── README.md  ← You are here
```

---

## Quick Reference

### Dataset Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Diseases** | 178 | MeSH-standardized disease terms |
| **Metabolites** | 2,006 | HMDB metabolites with known disease associations |
| **Proteins** | 4,912 | Proteins associated with metabolites |
| **GO Terms** | 12,524 | Gene Ontology annotations |
| **Total Entities** | 19,442 | All nodes in hypergraph |
| **Known Associations** | 4,000 | Positive disease-metabolite pairs |

### Data Sources

| Database | Version | Access Date | URL |
|----------|---------|-------------|-----|
| HMDB | 5.0 | March 15, 2024 | https://hmdb.ca |
| CTD | Update 2023 | April 10, 2024 | http://ctdbase.org |
| Gene Ontology | 2024 | March 2024 | http://geneontology.org |

---

## Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ RAW DATA (16 MB)                                             │
├─────────────────────────────────────────────────────────────┤
│ • HMDB XML parsing                                           │
│ • CTD data extraction                                        │
│ • Disease name standardization (MeSH)                        │
│ • Entity ID mapping                                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ SIMILARITY COMPUTATION (235 MB)                              │
├─────────────────────────────────────────────────────────────┤
│ • Metabolites: Tanimoto coefficient on SMILES (~2 hours)    │
│ • Proteins: BLAST+ alignment (~2-3 days)                    │
│ • GO Terms: Semantic similarity (~1 day)                    │
│ • Total computation time: ~3-4 days                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ HYPERGRAPH CONSTRUCTION (10 MB)                             │
├─────────────────────────────────────────────────────────────┤
│ • Build incidence matrix (178 × 19,442)                     │
│ • Map diseases as hyperedges                                │
│ • Connect metabolites, proteins, GO terms as nodes          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ HGNN TRAINING (180 MB)                                       │
├─────────────────────────────────────────────────────────────┤
│ • Autoencoder dimensionality reduction                       │
│ • 2-layer HGNN with message passing                         │
│ • 500-dimensional embeddings                                │
│ • Training time: ~1.5-2 days on GPU                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ CLASSIFICATION (See ../results/)                             │
├─────────────────────────────────────────────────────────────┤
│ • LightGBM training: ~8-10 minutes                          │
│ • 5-fold cross-validation                                   │
│ • Final performance: MCC=0.9305, AUC=0.9983                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Computational Cost Summary

| Processing Step | Time | Hardware | Can Skip? |
|----------------|------|----------|-----------|
| **Raw Data Extraction** | 1-2 days | CPU | ✅ Yes - provided |
| **Similarity Matrices** | 3-4 days | CPU (24 cores) | ✅ Yes - provided |
| **Hypergraph Construction** | 10 min | CPU | ✅ Yes - provided |
| **HGNN Training** | 1.5-2 days | GPU (RTX 4090) | ✅ Yes - embeddings provided |
| **LightGBM Training** | 8-10 min | CPU | ❌ No - must run for experiments |

**Total Time Saved by Using Provided Data:** ~5-7 days + GPU access

---

## File Formats

### Text Files (.txt)
- Encoding: UTF-8
- Line endings: LF (Unix-style)
- Numeric format: Space-separated, scientific notation where appropriate
- Missing values: Not applicable (all matrices are dense or complete)

### FASTA Files (.fasta)
- Standard FASTA format for protein sequences
- Header: `>HMDB_PID_XXXXX`
- Sequence: Amino acid one-letter codes

### Matrix Files
- Dense matrices stored as text with space-separated values
- Row-major order
- Symmetric matrices (similarity) stored in full (not upper/lower triangle only)
- Binary matrices (incidence) stored as sparse coordinate format (row, col, value)

---

## Data Quality and Limitations

### Assumptions
As stated in manuscript Methods Section 2.1:

> "Our analysis operates under the assumption that data retrieved from HMDB 5.0 and CTD are substantially accurate and complete. While these databases represent gold-standard resources undergoing continuous curation, they are not error-free. Previous studies identified potential inconsistencies, particularly in cross-referencing [Ref 20]."

### Known Limitations

1. **Database Incompleteness**
   - 4,000 known associations ≠ all true associations
   - Unknown associations may exist in literature
   - Negative samples may include false negatives

2. **Temporal Validity**
   - Data current as of March-April 2024
   - Newer associations may have been discovered since
   - Database updates may refine existing associations

3. **Cross-Reference Errors**
   - HMDB ↔ CTD mapping may contain errors
   - Disease name standardization to MeSH may lose nuance
   - Protein ID mapping across databases may be imperfect

4. **Computational Approximations**
   - BLAST E-value threshold (10) may miss remote homologs
   - Tanimoto similarity depends on fingerprint choice (Morgan, radius=2)
   - GO semantic similarity algorithm choice (Resnik) vs. alternatives

### Quality Control Performed

✅ Duplicate removal across HMDB and CTD
✅ Disease name standardization to MeSH
✅ SMILES structure validation (RDKit)
✅ Protein sequence format verification
✅ GO term existence check against official GO database
✅ Hypergraph connectivity verification (no isolated nodes)

---

## Reproducibility Guidelines

### Level 1: Verify Published Results
**Time:** 1 hour
**Requirements:** Provided embeddings + code
**Steps:**
1. Load `embeddings/*.txt`
2. Run `train_lightgbm.py` with provided config
3. Compare results with Tables 2-4 in manuscript

### Level 2: Reproduce HGNN Training
**Time:** 2 days
**Requirements:** GPU (RTX 4090 or equivalent) + all provided data
**Steps:**
1. Use provided `similarity_matrices/` and `hypergraph_structure/`
2. Run `train_hgnn.py` with same hyperparameters
3. Compare embeddings with provided `embeddings/*.txt`

### Level 3: Full Reproduction from Scratch
**Time:** 1 week
**Requirements:** High-performance workstation + all software
**Steps:**
1. Download fresh HMDB, CTD, GO data
2. Run all preprocessing scripts
3. Compute similarity matrices (~3-4 days)
4. Train HGNN (~2 days)
5. Train classifiers (~10 minutes)

**Expected Differences:**
- Newer databases → slightly different association counts
- Different random seeds → slightly different embeddings
- Overall performance metrics should be within ±1% of reported values

---

## Citation

```bibtex
@article{Xiao2025DHG-LGB,
  title={Identifying Metabolite-Disease Associations via Messaging in Hypergraphs},
  author={Xiao, F. and Ran, Y. and Li, Z.},
  journal={Metabolites},
  year={2025}
}
```

**Database Citations:**
- HMDB: Wishart DS, et al. Nucleic Acids Res. 2022.
- CTD: Davis AP, et al. Nucleic Acids Res. 2023.
- Gene Ontology: Carbon S, et al. Nucleic Acids Res. 2021.

---

## Contact

For questions about data processing or file formats:
- Refer to individual README files in subdirectories
- Check manuscript Methods Section 2
- Contact corresponding author (see manuscript)

---

**Last Updated:** January 10, 2025
**Data Version:** v2.0.0 - Complete dataset with all raw, processed, and learned representations
