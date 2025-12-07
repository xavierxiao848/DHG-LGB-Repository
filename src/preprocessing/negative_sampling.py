"""
Negative sample generation with indirect association filtering.

This module implements the negative sampling strategy described in Methods 2.2,
which excludes metabolite-disease pairs that share proteins to avoid false negatives.

Key Principle:
--------------
If a metabolite M associates with protein P, and protein P associates with disease D,
then the pair (M, D) should NOT be included in negative samples, even if no direct
M-D association is recorded in the database. This is because M and D are indirectly
associated through their shared protein P.

References
----------
See paper Methods section 2.2: "Negative Sampling Strategy"
"""

import numpy as np
import random
from typing import List, Tuple, Set
from tqdm import tqdm


def load_associations(
    associations_file: str
) -> Tuple[Set[Tuple[int, int]], Set[int], Set[int]]:
    """
    Load known metabolite-disease associations.

    Parameters
    ----------
    associations_file : str
        Path to associations file (format: metabolite_id disease_id per line)

    Returns
    -------
    positive_pairs : set of tuple
        Set of known (metabolite_idx, disease_idx) pairs
    metabolite_ids : set of int
        Set of all metabolite IDs
    disease_ids : set of int
        Set of all disease IDs
    """
    positive_pairs = set()
    metabolite_ids = set()
    disease_ids = set()

    with open(associations_file, 'r') as f:
        for line in f:
            metabolite_id, disease_id = line.strip().split()
            metabolite_id = int(metabolite_id)
            disease_id = int(disease_id)

            positive_pairs.add((metabolite_id, disease_id))
            metabolite_ids.add(metabolite_id)
            disease_ids.add(disease_id)

    return positive_pairs, metabolite_ids, disease_ids


def build_indirect_associations(
    metabolite_protein_file: str,
    protein_disease_file: str
) -> Set[Tuple[int, int]]:
    """
    Build set of indirectly associated metabolite-disease pairs (via shared proteins).

    For each protein P:
    - Find all metabolites M that associate with P
    - Find all diseases D that associate with P
    - Mark all (M, D) pairs as indirectly associated

    Parameters
    ----------
    metabolite_protein_file : str
        Path to metabolite-protein associations
        Format: metabolite_id protein_id per line
    protein_disease_file : str
        Path to protein-disease associations
        Format: protein_id disease_id per line

    Returns
    -------
    indirect_pairs : set of tuple
        Set of (metabolite_idx, disease_idx) pairs that are indirectly associated

    Examples
    --------
    >>> # If M1 associates with P1, and P1 associates with D1,
    >>> # then (M1, D1) is indirectly associated and should be excluded
    >>> # from negative samples
    >>> indirect = build_indirect_associations('met_prot.txt', 'prot_dis.txt')
    >>> print(f"Found {len(indirect)} indirect associations to exclude")

    Notes
    -----
    This filtering strategy is crucial for avoiding false negatives in the training
    data. Without it, many "negative" samples would actually be true associations
    mediated by proteins, leading to incorrect model training.

    Computational Complexity:
    - Time: O(n_proteins * avg_metabolites_per_protein * avg_diseases_per_protein)
    - Space: O(n_indirect_pairs)

    For the DHG-LGB dataset:
    - ~64,110 metabolite-protein associations
    - ~63,206 protein-GO associations (used to infer protein-disease links)
    - Results in substantial reduction of candidate negative pairs
    """
    # Load metabolite-protein associations
    metabolite_to_proteins = {}
    with open(metabolite_protein_file, 'r') as f:
        for line in f:
            metabolite_id, protein_id = line.strip().split()
            metabolite_id = int(metabolite_id)
            protein_id = int(protein_id)

            if metabolite_id not in metabolite_to_proteins:
                metabolite_to_proteins[metabolite_id] = set()
            metabolite_to_proteins[metabolite_id].add(protein_id)

    # Load protein-disease associations
    protein_to_diseases = {}
    with open(protein_disease_file, 'r') as f:
        for line in f:
            protein_id, disease_id = line.strip().split()
            protein_id = int(protein_id)
            disease_id = int(disease_id)

            if protein_id not in protein_to_diseases:
                protein_to_diseases[protein_id] = set()
            protein_to_diseases[protein_id].add(disease_id)

    # Build indirect associations
    indirect_pairs = set()
    for metabolite_id, proteins in tqdm(metabolite_to_proteins.items(),
                                         desc="Building indirect associations"):
        for protein_id in proteins:
            if protein_id in protein_to_diseases:
                for disease_id in protein_to_diseases[protein_id]:
                    indirect_pairs.add((metabolite_id, disease_id))

    return indirect_pairs


def generate_negative_samples(
    positive_pairs: Set[Tuple[int, int]],
    metabolite_ids: Set[int],
    disease_ids: Set[int],
    indirect_pairs: Set[Tuple[int, int]],
    ratio: float = 1.0,
    random_seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Generate negative samples with indirect association filtering.

    Negative Sample Selection Criteria:
    1. NOT in positive_pairs (no known direct association)
    2. NOT in indirect_pairs (no shared protein pathway)
    3. Randomly selected to match specified ratio

    Parameters
    ----------
    positive_pairs : set of tuple
        Known positive (metabolite, disease) associations
    metabolite_ids : set of int
        All metabolite IDs
    disease_ids : set of int
        All disease IDs
    indirect_pairs : set of tuple
        Indirectly associated pairs (via shared proteins) to exclude
    ratio : float, default=1.0
        Ratio of negative to positive samples
        - ratio=1.0: balanced dataset (equal positives and negatives)
        - ratio=2.0: twice as many negatives as positives
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    negative_samples : list of tuple
        List of (metabolite_id, disease_id) negative sample pairs

    Examples
    --------
    >>> positive = {(0, 0), (1, 2), (2, 1)}
    >>> indirect = {(0, 1), (1, 1)}  # Exclude these
    >>> metabolites = {0, 1, 2}
    >>> diseases = {0, 1, 2}
    >>> negatives = generate_negative_samples(
    ...     positive, metabolites, diseases, indirect, ratio=1.0
    ... )
    >>> print(f"Generated {len(negatives)} negative samples")

    Notes
    -----
    Dataset Balance Consideration:
    - Paper uses 1:1 ratio (4,000 positives + 4,000 negatives = 8,000 total)
    - Balanced datasets prevent bias toward majority class
    - Alternative ratios (1:2, 1:5, 1:10) tested in robustness analysis (Section 3.2)

    False Negative Risk:
    - Even with indirect filtering, some false negatives may remain
    - Associations mediated by other pathways (not captured in database) could exist
    - This is a fundamental limitation acknowledged in Discussion (Fourth limitation)
    """
    random.seed(random_seed)

    # Determine target number of negative samples
    n_positive = len(positive_pairs)
    n_negative_target = int(n_positive * ratio)

    # Build candidate pool (all possible pairs minus positive and indirect)
    excluded_pairs = positive_pairs | indirect_pairs

    candidate_pool = []
    for metabolite_id in metabolite_ids:
        for disease_id in disease_ids:
            if (metabolite_id, disease_id) not in excluded_pairs:
                candidate_pool.append((metabolite_id, disease_id))

    # Check if we have enough candidates
    if len(candidate_pool) < n_negative_target:
        print(f"WARNING: Only {len(candidate_pool)} candidate negative pairs available")
        print(f"         Requested {n_negative_target} negatives (ratio={ratio})")
        print(f"         Using all available candidates")
        return candidate_pool

    # Randomly sample from candidate pool
    negative_samples = random.sample(candidate_pool, n_negative_target)

    print(f"\nNegative Sampling Summary:")
    print(f"  Positive samples: {n_positive}")
    print(f"  Negative samples generated: {len(negative_samples)}")
    print(f"  Ratio (neg:pos): {len(negative_samples)/n_positive:.2f}")
    print(f"  Total excluded pairs: {len(excluded_pairs)}")
    print(f"    - Direct associations: {len(positive_pairs)}")
    print(f"    - Indirect associations: {len(indirect_pairs)}")

    return negative_samples


def save_samples(
    positive_pairs: Set[Tuple[int, int]],
    negative_samples: List[Tuple[int, int]],
    output_file: str
) -> None:
    """
    Save positive and negative samples to file.

    Output Format:
    metabolite_id disease_id label

    where label = 1 for positive, label = 0 for negative

    Parameters
    ----------
    positive_pairs : set of tuple
        Positive sample pairs
    negative_samples : list of tuple
        Negative sample pairs
    output_file : str
        Path to output file

    Examples
    --------
    >>> save_samples(positive_pairs, negative_samples, 'data/processed/samples.txt')
    """
    with open(output_file, 'w') as f:
        # Write positive samples
        for metabolite_id, disease_id in positive_pairs:
            f.write(f"{metabolite_id} {disease_id} 1\n")

        # Write negative samples
        for metabolite_id, disease_id in negative_samples:
            f.write(f"{metabolite_id} {disease_id} 0\n")

    print(f"\nSamples saved to: {output_file}")
    print(f"  Total samples: {len(positive_pairs) + len(negative_samples)}")


def main():
    """
    Example usage of negative sampling pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Generate negative samples with indirect filtering')
    parser.add_argument('--associations', required=True,
                        help='Path to positive associations file')
    parser.add_argument('--met-prot', required=True,
                        help='Path to metabolite-protein associations')
    parser.add_argument('--prot-dis', required=True,
                        help='Path to protein-disease associations')
    parser.add_argument('--output', required=True,
                        help='Path to output file')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Negative:positive ratio (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    print("Loading associations...")
    positive_pairs, metabolite_ids, disease_ids = load_associations(args.associations)

    print("Building indirect associations...")
    indirect_pairs = build_indirect_associations(args.met_prot, args.prot_dis)

    print("Generating negative samples...")
    negative_samples = generate_negative_samples(
        positive_pairs,
        metabolite_ids,
        disease_ids,
        indirect_pairs,
        ratio=args.ratio,
        random_seed=args.seed
    )

    save_samples(positive_pairs, negative_samples, args.output)
    print("\nDone!")


if __name__ == '__main__':
    main()
