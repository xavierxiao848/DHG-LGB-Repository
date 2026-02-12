# Similarity Matrices - Precomputed Entity Similarities

This directory contains three large precomputed similarity matrices that quantify pairwise similarity between biological entities. These matrices serve as initial node features for the Hypergraph Neural Network (HGNN).

**Total Size:** 235 MB
**Computation Time:** Approximately 3-4 days on high-performance workstation
**Manuscript Reference:** Methods Section 2.5

---

## Why Similarity Matrices Are Critical

**Computational Cost Savings:**
- Without these matrices, researchers must recompute ~24 million protein alignments (2-3 days)
- BLAST alignment requires specialized bioinformatics software and expertise
- Providing precomputed matrices ensures immediate reproducibility

**Scientific Transparency:**
- Exact similarity values used in experiments
- Enables verification of downstream analyses
- Allows investigation of feature importance

---

## Files Description

### 1. metabolite_similarity_2006x2006.txt (25 MB)

**Dimensions:** 2,006 × 2,006 (symmetric matrix)
**Computation Method:** Tanimoto coefficient on molecular fingerprints
**Value Range:** [0, 1], where 0 = completely dissimilar, 1 = identical
**Manuscript Reference:** Methods Section 2.5.1

**Computation Details:**

1. **Input:** SMILES chemical structures from `../raw/HMDB_SMILES_structures.txt`

2. **Fingerprint Generation:**
   - Tool: RDKit (version 2022.09.5)
   - Fingerprint type: Morgan fingerprints (radius=2, 2048 bits)
   - Equivalent to Extended-Connectivity Fingerprints (ECFP4)

3. **Tanimoto Coefficient Formula:**
   ```
   Tc(A, B) = |A ∩ B| / |A ∪ B|

   Where:
   A, B = molecular fingerprint bit vectors
   |A ∩ B| = count of bits set to 1 in both A and B
   |A ∪ B| = count of bits set to 1 in either A or B
   ```

4. **Example Computation:**
   ```python
   from rdkit import Chem, DataStructs
   from rdkit.Chem import AllChem

   # Convert SMILES to molecule
   mol1 = Chem.MolFromSmiles("C(C(=O)O)N")  # Glycine
   mol2 = Chem.MolFromSmiles("CC(C(=O)O)N")  # Alanine

   # Generate Morgan fingerprints
   fp1 = AllChem.GetMorganFingerprint(mol1, radius=2)
   fp2 = AllChem.GetMorganFingerprint(mol2, radius=2)

   # Compute Tanimoto similarity
   similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
   # Result: ~0.73 (high similarity, both are amino acids)
   ```

**Matrix Format:**
- Text file, space-separated values
- Row i, Column j = similarity between metabolite i and metabolite j
- Diagonal values = 1.0 (self-similarity)
- Symmetric matrix: similarity(i,j) = similarity(j,i)

**Computational Time:** ~2 hours on Intel i9-13900K (24 cores)

**Biological Interpretation:**
- High Tanimoto (>0.7): Structurally similar metabolites, likely similar biological functions
- Medium Tanimoto (0.3-0.7): Some shared substructures, potentially related pathways
- Low Tanimoto (<0.3): Structurally dissimilar, likely different chemical classes

---

### 2. protein_similarity_4912x4912.txt (60 MB)

**Dimensions:** 4,912 × 4,912 (symmetric matrix)
**Computation Method:** BLAST+ sequence alignment with BLOSUM62 scoring matrix
**Value Range:** [0, 1], normalized alignment scores
**Manuscript Reference:** Methods Section 2.5.2

**Computation Details:**

1. **Input:** Protein amino acid sequences from `../raw/protein_sequences.fasta`

2. **BLAST Parameters:**
   - Tool: BLAST+ version 2.13.0 (`blastp` program)
   - Scoring matrix: BLOSUM62
   - Gap opening penalty: 11
   - Gap extension penalty: 1
   - E-value threshold: 10
   - Word size: 3 (default)

3. **Pairwise Alignment:**
   - Total alignments: 4,912 × 4,912 = 24,127,744 pairs
   - Each alignment produces:
     - Bit score (unnormalized similarity score)
     - E-value (statistical significance)
     - Percent identity
     - Alignment length

4. **Normalization to [0,1]:**
   ```
   From manuscript Methods Section 2.5.2:
   "BLAST alignment scores were converted to normalized similarity values
   in [0, 1] using the transformation s = (identity_percentage) / 100,
   where identity_percentage represents the proportion of identical amino
   acid residues in the alignment."

   Formula:
   normalized_similarity = percent_identity / 100

   Alternative (if using bit scores):
   normalized_similarity = (bit_score - min_score) / (max_score - min_score)
   ```

5. **Example BLAST Output:**
   ```
   Query: HMDB_PID_001 (342 amino acids)
   Subject: HMDB_PID_002 (356 amino acids)

   Alignment:
   Query  1    MKVLWAALLVTFLAGCQAKVEQAVETNAE... 342
               ||||||||||||||||||||||||||||||
   Sbjct  1    MKVLWAALLVTFLAGCQAKVEQAVETNAE... 342

   Identities: 315/342 (92%)
   Bit score: 645
   E-value: 0.0

   Normalized similarity: 0.92
   ```

**Matrix Format:**
- Text file, space-separated values
- Row i, Column j = sequence similarity between protein i and protein j
- Diagonal values = 1.0 (self-similarity)
- Symmetric matrix (similarity is bidirectional)

**Computational Time:** ~2-3 days on Intel i9-13900K (24 cores, parallelized BLAST)

**Biological Interpretation:**
- High similarity (>0.70): Likely homologous proteins, similar structure/function
- Medium similarity (0.30-0.70): May share functional domains or motifs
- Low similarity (<0.30): Sequence-level dissimilarity, but may share higher-order features

**Performance Optimization:**
- Parallelization across 24 CPU cores using GNU Parallel
- Database indexing with `makeblastdb`
- Output format optimization (`-outfmt 6` for tabular output)

---

### 3. GO_similarity_12524x12524.txt (150 MB)

**Dimensions:** 12,524 × 12,524 (symmetric matrix)
**Computation Method:** Semantic similarity based on Gene Ontology DAG structure
**Value Range:** [0, 1], where higher values indicate closer functional relationship
**Manuscript Reference:** Methods Section 2.5.3

**Computation Details:**

1. **Input:** GO term hierarchy from `../raw/GO_ancestors_DAG.txt`

2. **Semantic Similarity Algorithm:**
   - Method: Resnik's semantic similarity (information content-based)
   - Reference: Resnik P. "Semantic similarity in a taxonomy." IJCAI 1999.

3. **Algorithm Steps:**

   **Step 1: Compute Information Content (IC)**
   ```
   For each GO term t:
     IC(t) = -log(P(t))

   Where P(t) = frequency of term t in annotations

   More specific terms (lower in DAG) → higher IC
   More general terms (higher in DAG) → lower IC
   ```

   **Step 2: Find Most Informative Common Ancestor (MICA)**
   ```
   For GO terms A and B:
     MICA(A, B) = argmax_{t ∈ ancestors(A) ∩ ancestors(B)} IC(t)

   The common ancestor with highest information content
   ```

   **Step 3: Calculate Semantic Similarity**
   ```
   Resnik similarity:
     sim(A, B) = IC(MICA(A, B))

   Normalized to [0, 1]:
     normalized_sim(A, B) = IC(MICA(A, B)) / max_IC

   Where max_IC = maximum information content across all terms
   ```

4. **Example Calculation:**
   ```
   GO:0008150 (biological_process) - Very general, IC = 0.5
   GO:0044237 (cellular metabolic process) - More specific, IC = 3.2
   GO:0006096 (glycolytic process) - Very specific, IC = 8.1

   Similarity between two glycolysis-related terms:
   - Both descendants of GO:0006096
   - MICA = GO:0006096, IC = 8.1
   - Normalized similarity ≈ 0.95 (very similar)

   Similarity between glycolysis and cell division:
   - MICA = GO:0008150 (biological_process), IC = 0.5
   - Normalized similarity ≈ 0.12 (not very similar)
   ```

**Matrix Format:**
- Text file, space-separated values
- Row i, Column j = semantic similarity between GO term i and GO term j
- Diagonal values = 1.0 (self-similarity)
- Symmetric matrix

**Computational Time:** ~1 day using graph traversal algorithms on GO DAG

**Biological Interpretation:**
- High similarity (>0.8): GO terms in same pathway or closely related functions
- Medium similarity (0.4-0.8): Related but distinct biological processes
- Low similarity (<0.4): Functionally distant processes

**Software Used:**
- GOSemSim R package (version 2.22.0)
- Or custom Python implementation using NetworkX for DAG traversal

---

## Rationale for Similarity-Based Features

As explained in manuscript Methods Section 2.5.5 "Rationale for Similarity Matrix Representation":

**1. Transductive Learning Framework Necessity:**
> "The HGNN operates in a transductive setting where the entire graph structure
> (all metabolites, diseases, proteins, and GO terms) must be known during training.
> The model cannot make predictions for completely unseen metabolites that were
> not part of the original hypergraph."

**2. Biological Consistency Principle:**
> "Structurally or functionally similar entities tend to exhibit similar biological
> behaviors. In metabolite-disease contexts, metabolites with similar chemical
> structures often associate with similar diseases."

**3. Domain-Specific Metrics:**
> "Similarity matrices leverage precomputed domain-specific measures (Tanimoto
> for chemical structures, BLAST for sequences, semantic similarity for GO DAG)
> that incorporate decades of domain knowledge."

**4. Computational Efficiency:**
> "Computing these similarities once and using them as features is more efficient
> than requiring HGNN to learn equivalent similarity functions from raw data."

---

## Min-Max Normalization

**From manuscript Methods Section 2.5.6:**

All three similarity matrices undergo Min-Max normalization to [0, 1] to address scale incompatibilities:

```python
def min_max_normalize(similarity_matrix):
    """
    Normalize similarity matrix to [0, 1] range
    """
    min_val = similarity_matrix.min()
    max_val = similarity_matrix.max()

    normalized = (similarity_matrix - min_val) / (max_val - min_val)

    return normalized
```

**Purpose:**
- Ensure all three similarity types are on comparable scales
- Prevent any single similarity type from dominating HGNN learning
- Enable direct comparison of similarity values across entity types

---

## Usage in HGNN

These similarity matrices serve as **initial node features** in the Hypergraph Neural Network:

```python
# Pseudocode from Methods Section 2.6
X_metabolites = metabolite_similarity_matrix  # 2006 × 2006
X_proteins = protein_similarity_matrix        # 4912 × 4912
X_GO = GO_similarity_matrix                   # 12524 × 12524

# Dimensionality reduction via autoencoders
X_metabolites_reduced = autoencoder(X_metabolites)  # 2006 → 500
X_proteins_reduced = autoencoder(X_proteins)        # 4912 → 500
X_GO_reduced = autoencoder(X_GO)                    # 12524 → 500

# HGNN message passing
for layer in HGNN_layers:
    X = propagate_via_hypergraph(X, hypergraph_structure)
    X = apply_nonlinearity(X)

# Output: learned 500-dim embeddings (see ../embeddings/)
```

---

## File Statistics

| File | Size | Dimensions | Non-Zero Entries | Sparsity |
|------|------|------------|------------------|----------|
| metabolite_similarity_2006x2006.txt | 25 MB | 2,006 × 2,006 | ~4,024,036 | Dense |
| protein_similarity_4912x4912.txt | 60 MB | 4,912 × 4,912 | ~24,127,744 | Dense |
| GO_similarity_12524x12524.txt | 150 MB | 12,524 × 12,524 | ~156,850,576 | Dense |

**Total:** 235 MB, representing ~185 million pairwise similarity values

---

## Reproducibility Instructions

### To Recompute Metabolite Similarity Matrix:

```bash
# Install RDKit
conda install -c conda-forge rdkit

# Python script
python compute_metabolite_similarity.py \
    --smiles ../raw/HMDB_SMILES_structures.txt \
    --output metabolite_similarity_2006x2006.txt \
    --fingerprint morgan \
    --radius 2 \
    --nbits 2048
```

### To Recompute Protein Similarity Matrix:

```bash
# Install BLAST+
conda install -c bioconda blast

# Create BLAST database
makeblastdb -in ../raw/protein_sequences.fasta \
            -dbtype prot \
            -out protein_db

# Run all-vs-all BLAST
blastp -query ../raw/protein_sequences.fasta \
       -db protein_db \
       -out blast_results.txt \
       -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
       -num_threads 24 \
       -matrix BLOSUM62 \
       -gapopen 11 \
       -gapextend 1 \
       -evalue 10

# Convert to similarity matrix
python parse_blast_to_matrix.py \
    --blast blast_results.txt \
    --output protein_similarity_4912x4912.txt
```

### To Recompute GO Similarity Matrix:

```R
# Install GOSemSim
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GOSemSim")

# R script
library(GOSemSim)
library(GO.db)

# Load GO terms
go_data <- godata('org.Hs.eg.db', ont="BP")

# Compute pairwise semantic similarity
similarity_matrix <- mgoSim(go_terms, go_terms,
                            semData=go_data,
                            measure="Resnik",
                            combine=NULL)

write.table(similarity_matrix,
            "GO_similarity_12524x12524.txt",
            row.names=FALSE, col.names=FALSE)
```

**Expected Time:** 3-4 days total for all three matrices

---

## Citation

If you use these similarity matrices, please cite:

**Methods:**
- Tanimoto: Bajusz D, et al. "Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?" J Cheminform. 2015.
- BLAST: Altschul SF, et al. "Gapped BLAST and PSI-BLAST." Nucleic Acids Res. 1997.
- GO Semantic Similarity: Yu G, et al. "GOSemSim: an R package for measuring semantic similarity among GO terms and gene products." Bioinformatics. 2010.

**Our Manuscript:**
```bibtex
@article{Xiao2025DHG-LGB,
  title={Identifying Metabolite-Disease Associations via Messaging in Hypergraphs},
  author={Xiao, F. and Ran, Y. and Li, Z.},
  journal={Metabolites},
  year={2025}
}
```

---

**Last Updated:** January 10, 2025
**Matrix Version:** Computed from HMDB 5.0, UniProt 2024, GO 2024 releases
