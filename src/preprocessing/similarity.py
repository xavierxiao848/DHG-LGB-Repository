"""
Similarity matrix computation for metabolites, proteins, and GO terms.

This module implements three types of similarity computations as described in the paper:
1. Metabolite similarity: Tanimoto coefficient based on Morgan fingerprints
2. Protein similarity: BLAST sequence alignment with BLOSUM62 matrix
3. GO semantic similarity: Ancestral contribution method

References
----------
.. [1] Altschul, S. F., et al. (1990). Basic local alignment search tool.
       Journal of Molecular Biology, 215(3), 403-410.
.. [2] Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints.
       Journal of Chemical Information and Modeling, 50(5), 742-754.
"""

import numpy as np
from typing import Dict, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def compute_tanimoto_similarity(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048
) -> np.ndarray:
    """
    Compute Tanimoto similarity matrix for metabolites based on Morgan fingerprints.

    The Tanimoto coefficient (also called Jaccard index) measures structural similarity
    between molecules by comparing their molecular fingerprints:

    Tc(A, B) = |A ∩ B| / |A ∪ B|

    where |A ∩ B| is the number of common features and |A ∪ B| is the total number
    of unique features in either molecule.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings representing molecular structures
    radius : int, default=2
        Radius for Morgan fingerprint (equivalent to ECFP4 when radius=2)
    n_bits : int, default=2048
        Number of bits in the fingerprint vector

    Returns
    -------
    np.ndarray
        Similarity matrix (shape: [n_metabolites, n_metabolites])
        Values range from 0 (no similarity) to 1 (identical structures)

    Examples
    --------
    >>> smiles = ['CCO', 'CC(O)C', 'CCCO']  # Ethanol, Isopropanol, Propanol
    >>> sim_matrix = compute_tanimoto_similarity(smiles)
    >>> print(f"Similarity matrix shape: {sim_matrix.shape}")

    Notes
    -----
    Morgan fingerprints (also known as circular fingerprints or ECFP) encode
    structural features by considering atom environments up to a specified radius.
    The Tanimoto coefficient is naturally bounded in [0, 1], requiring no additional
    normalization.

    Implementation Details:
    1. Convert SMILES to RDKit molecule objects
    2. Generate Morgan fingerprints using AllChem.GetMorganFingerprintAsBitVect()
    3. Compute pairwise Tanimoto similarity using DataStructs.TanimotoSimilarity()

    For detailed algorithms, see RDKit documentation:
    https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
    """
    n_metabolites = len(smiles_list)
    similarity_matrix = np.zeros((n_metabolites, n_metabolites))

    # Generate fingerprints
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(fp)
        else:
            # Handle invalid SMILES (use zero fingerprint)
            fingerprints.append(None)

    # Compute pairwise similarities
    for i in range(n_metabolites):
        for j in range(i, n_metabolites):
            if fingerprints[i] is not None and fingerprints[j] is not None:
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            else:
                similarity = 0.0

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix

    return similarity_matrix


def compute_blast_similarity(
    fasta_file: str,
    blast_program: str = 'blastp',
    matrix: str = 'BLOSUM62',
    gap_open: int = 11,
    gap_extend: int = 1,
    evalue_threshold: float = 10.0
) -> np.ndarray:
    """
    Compute protein sequence similarity using BLAST alignment.

    BLAST (Basic Local Alignment Search Tool) identifies regions of local similarity
    between protein sequences. Similarity scores are based on sequence identity
    percentage, normalized to [0, 1] range.

    Parameters
    ----------
    fasta_file : str
        Path to FASTA file containing protein sequences
    blast_program : str, default='blastp'
        BLAST program (blastp for protein-protein alignment)
    matrix : str, default='BLOSUM62'
        Scoring matrix (BLOSUM62 is standard for protein comparison)
    gap_open : int, default=11
        Gap opening penalty
    gap_extend : int, default=1
        Gap extension penalty
    evalue_threshold : float, default=10.0
        E-value threshold for reporting alignments

    Returns
    -------
    np.ndarray
        Similarity matrix (shape: [n_proteins, n_proteins])
        Values range from 0 (no similarity) to 1 (identical sequences)

    Notes
    -----
    BLAST Similarity Computation Pipeline:
    1. Build BLAST database from FASTA file using makeblastdb
    2. Perform all-vs-all pairwise alignments using blastp
    3. Extract sequence identity percentage from alignment results
    4. Normalize to [0, 1]: similarity = identity_percentage / 100

    BLOSUM62 Matrix:
    - Standard substitution matrix for protein sequence comparison
    - Based on observed amino acid substitutions in related proteins
    - Optimized for detecting sequences with ~62% identity

    Gap Penalties:
    - Gap open penalty: 11 (cost of introducing a gap)
    - Gap extend penalty: 1 (cost of extending an existing gap)
    - These values are standard for BLAST protein alignment

    Implementation Framework:
    ```python
    from Bio.Blast.Applications import NcbiblastpCommandline
    from Bio.Blast import NCBIXML

    # 1. Build BLAST database
    os.system(f"makeblastdb -in {fasta_file} -dbtype prot")

    # 2. Run BLAST for each sequence
    for query_seq in sequences:
        blast_cline = NcbiblastpCommandline(
            query=query_seq,
            db=fasta_file,
            evalue=evalue_threshold,
            outfmt=5,  # XML output
            matrix=matrix,
            gapopen=gap_open,
            gapextend=gap_extend
        )
        result = blast_cline()[0]

        # 3. Parse results and extract identity
        for record in NCBIXML.parse(result):
            for alignment in record.alignments:
                identity_pct = alignment.hsps[0].identities / alignment.hsps[0].align_length
                similarity = identity_pct / 100  # Normalize to [0, 1]
    ```

    For complete implementation, refer to BioPython documentation:
    https://biopython.org/docs/1.75/api/Bio.Blast.Applications.html
    """
    # Framework implementation
    # NOTE: Full BLAST computation is time-consuming (hours for 4,912 proteins)
    # This is a placeholder showing the structure

    print("BLAST Similarity Computation Framework")
    print(f"  Program: {blast_program}")
    print(f"  Matrix: {matrix}")
    print(f"  Gap penalties: open={gap_open}, extend={gap_extend}")
    print(f"  E-value threshold: {evalue_threshold}")
    print("\nNOTE: Full BLAST execution requires BioPython and NCBI BLAST+ tools")
    print("      For production use, implement the pipeline described in docstring")

    # Return placeholder (in production, compute actual BLAST similarities)
    # n_proteins = count_sequences(fasta_file)
    # return np.eye(n_proteins)  # Placeholder: identity matrix

    raise NotImplementedError(
        "BLAST similarity computation is computationally intensive. "
        "Please refer to the docstring for implementation framework."
    )


def compute_go_semantic_similarity(
    go_ancestors: Dict[str, List[str]],
    go_terms: List[str]
) -> np.ndarray:
    """
    Compute GO semantic similarity using ancestral contribution method.

    GO terms are organized in a Directed Acyclic Graph (DAG) where terms are connected
    by 'is_a' and 'part_of' relationships. Semantic similarity reflects functional
    relatedness based on shared ancestors in the GO DAG.

    The ancestral contribution method computes similarity as:

    sim(GO_i, GO_j) = |ancestors(GO_i) ∩ ancestors(GO_j)| / |ancestors(GO_i) ∪ ancestors(GO_j)|

    Parameters
    ----------
    go_ancestors : dict
        Dictionary mapping each GO term to its list of ancestors
        Format: {'GO:0006096': ['GO:0006091', 'GO:0008150', ...], ...}
    go_terms : list of str
        List of GO term IDs to compute similarities for

    Returns
    -------
    np.ndarray
        Similarity matrix (shape: [n_go_terms, n_go_terms])
        Values range from 0 (no shared ancestors) to 1 (identical terms)

    Examples
    --------
    >>> go_ancestors = {
    ...     'GO:0006096': ['GO:0006091', 'GO:0008150', 'GO:0044237'],
    ...     'GO:0006091': ['GO:0008150', 'GO:0044237'],
    ...     'GO:0008150': []  # Root term (biological process)
    ... }
    >>> go_terms = ['GO:0006096', 'GO:0006091']
    >>> sim_matrix = compute_go_semantic_similarity(go_ancestors, go_terms)

    Notes
    -----
    Ancestral Contribution Method:
    1. For each GO term, retrieve all ancestors from the GO DAG
    2. Compute Jaccard similarity between ancestor sets
    3. Similarity is naturally in [0, 1] range

    Alternative Methods (not used in this paper):
    - Resnik's method: Uses information content of most informative common ancestor
    - Lin's method: Normalizes Resnik's measure by information content of both terms
    - Wang's method: Considers the semantic contribution of all ancestors

    GO DAG Structure:
    - Root terms: 'biological_process', 'molecular_function', 'cellular_component'
    - Child terms inherit all ancestors from parent terms
    - Terms can have multiple parents (thus forming a DAG, not a tree)

    For GO DAG access, see:
    https://www.geneontology.org/docs/download-ontology/
    """
    n_terms = len(go_terms)
    similarity_matrix = np.zeros((n_terms, n_terms))

    for i in range(n_terms):
        for j in range(i, n_terms):
            go_i = go_terms[i]
            go_j = go_terms[j]

            # Get ancestor sets (include the term itself)
            ancestors_i = set(go_ancestors.get(go_i, [])) | {go_i}
            ancestors_j = set(go_ancestors.get(go_j, [])) | {go_j}

            # Compute Jaccard similarity (ancestral contribution)
            intersection = ancestors_i & ancestors_j
            union = ancestors_i | ancestors_j

            if len(union) > 0:
                similarity = len(intersection) / len(union)
            else:
                similarity = 0.0

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric

    return similarity_matrix


def normalize_similarity_matrix(
    similarity_matrix: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize similarity matrix to [0, 1] range.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Input similarity matrix
    method : str, default='minmax'
        Normalization method: 'minmax' or 'zscore'

    Returns
    -------
    np.ndarray
        Normalized similarity matrix
    """
    if method == 'minmax':
        min_val = similarity_matrix.min()
        max_val = similarity_matrix.max()
        if max_val > min_val:
            normalized = (similarity_matrix - min_val) / (max_val - min_val)
        else:
            normalized = similarity_matrix
    elif method == 'zscore':
        mean_val = similarity_matrix.mean()
        std_val = similarity_matrix.std()
        if std_val > 0:
            normalized = (similarity_matrix - mean_val) / std_val
        else:
            normalized = similarity_matrix
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized
