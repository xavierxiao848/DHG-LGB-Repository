# Raw Data - Original Database Extracts

This directory contains all raw data extracted from public databases used in the DHG-LGB study.

## Data Sources and Access Dates

All data were retrieved from publicly accessible databases during the manuscript preparation period:

| Database | Version | Access Date | Purpose |
|----------|---------|-------------|---------|
| HMDB | 5.0 | March 15, 2024 | Metabolite data, disease associations, protein information |
| CTD | Update 2023 | April 10, 2024 | Comparative toxicogenomics data, disease-metabolite associations |
| Gene Ontology | 2024 release | March 2024 | GO term hierarchy and annotations |

## Files Description

### 1. HMDB_data_raw.txt (5.7 MB)
**Source:** Human Metabolome Database (HMDB) version 5.0

**Content:**
- Metabolite-disease associations from HMDB
- Metabolite identifiers (HMDB IDs)
- Disease identifiers (OMIM IDs, MeSH terms)
- Protein-metabolite relationships
- Initial dataset: 2,347 metabolites, 1,024 diseases, 2,156 associations

**Format:** Tab-separated text file

**Extraction Method:**
- HMDB XML files downloaded and parsed
- Relevant metabolite-disease relationships extracted
- Disease names standardized to MeSH terminology

### 2. CTD_data_raw.txt (4.7 MB)
**Source:** Comparative Toxicogenomics Database (CTD) Update 2023

**Content:**
- Additional disease-metabolite associations not in HMDB
- Contributed 1,844 additional associations
- Cross-referenced with HMDB to avoid duplicates

**Format:** Tab-separated text file with CTD-specific identifiers

**Note:** Combined HMDB + CTD yielded 4,000 unique positive disease-metabolite associations used in the study

### 3. protein_sequences.fasta (3.0 MB)
**Source:** HMDB 5.0 (protein sequences associated with metabolites)

**Content:**
- Amino acid sequences for 4,912 proteins
- Protein identifiers (HMDB PIDs)
- Used for BLAST-based sequence similarity computation

**Format:** Standard FASTA format
```
>HMDB_PID_12345
MKVLWAALLVTFLAGCQAKVEQAVE...
```

**Usage:** Input for BLAST+ 2.13.0 protein similarity matrix computation

### 4. GO_ancestors_DAG.txt (2.6 MB)
**Source:** Gene Ontology Consortium (2024 release)

**Content:**
- Gene Ontology Directed Acyclic Graph (DAG) structure
- Parent-child relationships between GO terms
- Used for computing semantic similarity between GO annotations
- Covers 12,524 GO terms present in the dataset

**Format:** Edge list representing DAG structure

**Example:**
```
GO:0008150    GO:0009987    (biological_process -> cellular_process)
GO:0009987    GO:0044237    (cellular_process -> cellular metabolic process)
```

### 5. HMDB_SMILES_structures.txt (152 KB)
**Source:** HMDB 5.0

**Content:**
- SMILES (Simplified Molecular Input Line Entry System) chemical structures
- For 2,006 metabolites in the final dataset
- Used for Tanimoto coefficient-based similarity computation

**Format:** Two columns - HMDB ID and SMILES string

**Example:**
```
HMDB0000001    C(C(=O)O)N
HMDB0000002    CC(C(=O)O)N
```

**Note:** SMILES strings converted to molecular fingerprints using RDKit for similarity calculation

### 6. disease_to_entity_mapping.txt (2.4 MB)
**Source:** Derived from HMDB and CTD data integration

**Content:**
- Mapping between diseases and all associated biological entities
- Includes metabolites, proteins, and GO annotations linked to each disease
- Foundation for hypergraph construction (diseases as hyperedges)

**Format:** Disease ID followed by lists of connected entity IDs

**Structure:**
```
Disease_ID    Metabolite_IDs    Protein_IDs    GO_term_IDs
D003920       HMDB0000001,HMDB0000122,...    PID_345,PID_678,...    GO:0008150,GO:0009987,...
```

## Data Processing Pipeline

```
Raw Data Extraction
    ↓
[HMDB XML Parsing] + [CTD Data Integration]
    ↓
Entity Standardization (MeSH terms, HMDB IDs)
    ↓
Cross-Database Deduplication
    ↓
Final Dataset: 178 diseases, 2,006 metabolites, 4,912 proteins, 12,524 GO terms
    ↓
Processed Data (see ../processed/)
```

## Data Quality and Limitations

### Assumptions
As stated in the manuscript (Methods Section 2.1):

> "Our analysis operates under the assumption that data retrieved from HMDB 5.0 and CTD (update 2023) are substantially accurate and complete for the purposes of this study. While these databases represent gold-standard resources in metabolomics and toxicogenomics research, undergoing continuous curation and expert review, they are not error-free."

### Known Limitations
1. **Database Incompleteness**: Known positive associations represent only a subset of true biological relationships
2. **Cross-Reference Errors**: Previous studies identified potential inconsistencies in metabolic databases, particularly in cross-referencing [Ref 20 in manuscript]
3. **Temporal Validity**: Data reflect database status as of access dates (March-April 2024); newer associations may have been added subsequently

### Quality Control Steps Taken
- ✅ Duplicate removal across HMDB and CTD
- ✅ Disease name standardization to MeSH controlled vocabulary
- ✅ SMILES structure validation using RDKit
- ✅ Protein sequence format verification
- ✅ GO term existence verification against official GO database

## Reproducibility Notes

### To Extract Fresh Data from Current Databases:

1. **HMDB (latest version):**
   ```bash
   # Download HMDB XML files from https://hmdb.ca/downloads
   # Parse XML to extract metabolite-disease associations
   # Example parser available in code/ directory
   ```

2. **CTD (latest version):**
   ```bash
   # Download CTD data from http://ctdbase.org/downloads/
   # Extract chemical-disease associations
   # Cross-reference with HMDB metabolites
   ```

3. **Gene Ontology:**
   ```bash
   # Download GO DAG from http://geneontology.org/docs/download-ontology/
   # Use OBO format or RDF/XML format
   ```

**Expected Differences:**
- Newer database versions will contain additional associations
- Some associations may be deprecated or refined
- Overall methodology remains applicable

## File Statistics

| File | Size | Lines | Unique Entities |
|------|------|-------|-----------------|
| HMDB_data_raw.txt | 5.7 MB | ~45,000 | 2,347 metabolites, 1,024 diseases |
| CTD_data_raw.txt | 4.7 MB | ~38,000 | 1,844 additional associations |
| protein_sequences.fasta | 3.0 MB | ~95,000 | 4,912 proteins |
| GO_ancestors_DAG.txt | 2.6 MB | ~67,000 edges | 12,524 GO terms |
| HMDB_SMILES_structures.txt | 152 KB | 2,006 | 2,006 metabolites |
| disease_to_entity_mapping.txt | 2.4 MB | 178 diseases | All entities |

**Total:** 16.5 MB

## Citation

If you use this raw data, please cite both the original databases and our manuscript:

**Databases:**
- HMDB: Wishart DS, et al. HMDB 5.0: the Human Metabolome Database for 2022. Nucleic Acids Res. 2022.
- CTD: Davis AP, et al. Comparative Toxicogenomics Database (CTD): update 2023. Nucleic Acids Res. 2023.
- Gene Ontology: Carbon S, et al. The Gene Ontology resource: enriching a GOld mine. Nucleic Acids Res. 2021.

**Our Manuscript:**
```bibtex
@article{Xiao2025DHG-LGB,
  title={Identifying Metabolite-Disease Associations via Messaging in Hypergraphs},
  author={Xiao, F. and Ran, Y. and Li, Z.},
  journal={Metabolites},
  year={2025}
}
```

## Contact

For questions about data extraction or preprocessing:
- See code/ directory for extraction scripts
- Refer to manuscript Methods Section 2.1 for detailed methodology
- Contact corresponding author for specific queries

---

**Last Updated:** January 10, 2025
**Data Version:** Raw extracts from HMDB 5.0 (2024-03-15) and CTD 2023 (2024-04-10)
