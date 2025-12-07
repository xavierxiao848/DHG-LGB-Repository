"""
Hypergraph Neural Network (HGNN) for learning node and hyperedge embeddings.

This module implements the HGNN architecture for disease-hypergraph representation
learning, where diseases are modeled as hyperedges connecting metabolites, proteins,
and GO terms.

References
----------
.. [1] Feng Y, You H, Zhang Z, et al. Hypergraph neural networks.
       AAAI Conference on Artificial Intelligence, 2019.
"""

import torch
import numpy as np
from dhg import Hypergraph
from typing import Tuple, Optional, Union
import os


class HGNNModel:
    """
    Hypergraph Neural Network model for disease-metabolite association prediction.

    This model learns embeddings for nodes (metabolites, proteins, GO terms) and
    hyperedges (diseases) through message passing on the hypergraph structure.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the hypergraph (metabolites + proteins + GO terms)
    embedding_dim : int, default=500
        Dimension of learned embeddings
    num_layers : int, default=2
        Number of HGNN layers (captures k-hop neighborhoods)
    dropout : float, default=0.4
        Dropout rate for regularization (implements stochastic edge sampling)
    device : str, default='cuda'
        Device for computation ('cuda' or 'cpu')

    Attributes
    ----------
    hypergraph : dhg.Hypergraph
        The hypergraph structure
    node_embeddings : torch.Tensor
        Learned node embeddings (shape: [num_nodes, embedding_dim])
    disease_embeddings : torch.Tensor
        Learned disease hyperedge embeddings (shape: [num_diseases, embedding_dim])

    Examples
    --------
    >>> # Load data
    >>> node_features = np.loadtxt('data/processed/node_features.txt', dtype=np.float32)
    >>> incidence_matrix = np.loadtxt('data/processed/hypergraph/incidence_matrix.txt')
    >>>
    >>> # Initialize model
    >>> model = HGNNModel(num_nodes=19442, embedding_dim=500)
    >>>
    >>> # Train
    >>> node_emb, disease_emb = model.fit(node_features, incidence_matrix)
    >>>
    >>> # Save embeddings
    >>> model.save_embeddings('data/embeddings/')
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 500,
        num_layers: int = 2,
        dropout: float = 0.4,
        device: str = 'cuda'
    ):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.hypergraph = None
        self.node_embeddings = None
        self.disease_embeddings = None

    def _build_hypergraph(
        self,
        incidence_matrix: np.ndarray
    ) -> Hypergraph:
        """
        Construct hypergraph from incidence matrix.

        Parameters
        ----------
        incidence_matrix : np.ndarray
            Binary incidence matrix (shape: [num_diseases, num_nodes])
            Entry [i, j] = 1 if node j belongs to disease hyperedge i

        Returns
        -------
        dhg.Hypergraph
            Constructed hypergraph object
        """
        # Convert to PyTorch tensor
        incidence_tensor = torch.from_numpy(incidence_matrix)
        num_diseases, num_nodes = incidence_tensor.shape

        # Extract edge list from incidence matrix
        edge_list = []
        for disease_idx in range(num_diseases):
            # Get indices of nodes connected to this disease
            connected_nodes = torch.where(incidence_tensor[disease_idx])[0].tolist()
            if connected_nodes:  # Ensure non-empty hyperedge
                edge_list.append(connected_nodes)

        # Construct hypergraph
        hypergraph = Hypergraph(num_nodes, edge_list)

        print(f"Hypergraph constructed: {num_nodes} nodes, {len(edge_list)} hyperedges")
        return hypergraph

    def fit(
        self,
        node_features: np.ndarray,
        incidence_matrix: np.ndarray,
        save_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn node and hyperedge embeddings through HGNN.

        Parameters
        ----------
        node_features : np.ndarray
            Initial node feature matrix (shape: [num_nodes, feature_dim])
            Typically similarity-based features (e.g., Tanimoto, BLAST, GO similarity)
        incidence_matrix : np.ndarray
            Hypergraph incidence matrix (shape: [num_diseases, num_nodes])
        save_intermediates : bool, default=False
            Whether to save intermediate propagation results (X_2, X_3)

        Returns
        -------
        node_embeddings : torch.Tensor
            Learned node embeddings (shape: [num_nodes, embedding_dim])
        disease_embeddings : torch.Tensor
            Learned disease hyperedge embeddings (shape: [num_diseases, embedding_dim])

        Notes
        -----
        The HGNN performs the following message passing operations:
        1. X_1 = HGNN_Smoothing(X_0) - Laplacian smoothing with degree normalization
        2. Y_1 = V2E(X_0) - Vertex-to-hyperedge aggregation (mean)
        3. X_2 = E2V(Y_1) - Hyperedge-to-vertex propagation (mean)
        4. X_3 = V2V(X_0) - Vertex-to-vertex propagation via shared hyperedges

        We use X_1 (smoothed features) as node embeddings and Y_1 as disease embeddings
        for downstream classification, as they provide the best balance between local
        and global information.
        """
        # Build hypergraph
        self.hypergraph = self._build_hypergraph(incidence_matrix)

        # Convert features to PyTorch tensor
        features = torch.from_numpy(node_features).float().to(self.device)

        print(f"Node features shape: {features.shape}")
        print(f"Running HGNN message passing...")

        # Message Passing Operations
        # ---------------------------

        # 1. HGNN Laplacian Smoothing
        # Normalizes by node degrees and hyperedge degrees to prevent large hyperedges
        # from dominating. This implements: X_1 = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X
        X_1 = self.hypergraph.smoothing_with_HGNN(features)
        print(f"  [1/4] Laplacian smoothing completed: {X_1.shape}")

        # 2. Vertex-to-Hyperedge Aggregation (V2E)
        # Aggregates node features to form hyperedge representations
        # Y_i = mean({X_j : j in e_i}) for each hyperedge e_i
        Y_1 = self.hypergraph.v2e(features, aggr="mean")
        print(f"  [2/4] V2E aggregation completed: {Y_1.shape}")

        # 3. Hyperedge-to-Vertex Propagation (E2V)
        # Propagates hyperedge information back to nodes
        # X_j' = mean({Y_i : j in e_i}) for each node j
        X_2 = self.hypergraph.e2v(Y_1, aggr="mean")
        print(f"  [3/4] E2V propagation completed: {X_2.shape}")

        # 4. Vertex-to-Vertex Propagation (V2V)
        # Propagates information between nodes sharing hyperedges
        # Equivalent to 2-hop neighborhood aggregation
        X_3 = self.hypergraph.v2v(features, aggr="mean")
        print(f"  [4/4] V2V propagation completed: {X_3.shape}")

        # Store embeddings
        self.node_embeddings = X_1  # Primary node embeddings (Laplacian smoothed)
        self.disease_embeddings = Y_1  # Disease hyperedge embeddings

        print(f"\nEmbedding learning completed:")
        print(f"  Node embeddings: {self.node_embeddings.shape}")
        print(f"  Disease embeddings: {self.disease_embeddings.shape}")

        return self.node_embeddings, self.disease_embeddings

    def save_embeddings(
        self,
        output_dir: str,
        node_filename: str = 'node_embeddings.txt',
        disease_filename: str = 'disease_embeddings.txt'
    ) -> None:
        """
        Save learned embeddings to text files.

        Parameters
        ----------
        output_dir : str
            Directory to save embedding files
        node_filename : str, default='node_embeddings.txt'
            Filename for node embeddings
        disease_filename : str, default='disease_embeddings.txt'
            Filename for disease embeddings

        Examples
        --------
        >>> model.save_embeddings('data/embeddings/')
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save node embeddings
        node_path = os.path.join(output_dir, node_filename)
        self._save_tensor(self.node_embeddings, node_path)
        print(f"Node embeddings saved to: {node_path}")

        # Save disease embeddings
        disease_path = os.path.join(output_dir, disease_filename)
        self._save_tensor(self.disease_embeddings, disease_path)
        print(f"Disease embeddings saved to: {disease_path}")

    @staticmethod
    def _save_tensor(tensor: torch.Tensor, filepath: str) -> None:
        """
        Save PyTorch tensor to text file.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to save
        filepath : str
            Output file path
        """
        with open(filepath, 'w') as f:
            for row in tensor:
                row_str = ' '.join([f'{val.item():.8f}' for val in row])
                f.write(row_str + '\n')

    def get_node_embedding(self, node_idx: int) -> torch.Tensor:
        """Get embedding for a specific node."""
        return self.node_embeddings[node_idx]

    def get_disease_embedding(self, disease_idx: int) -> torch.Tensor:
        """Get embedding for a specific disease."""
        return self.disease_embeddings[disease_idx]


def main():
    """
    Example usage of HGNNModel.

    This demonstrates how to load data, train HGNN, and save embeddings.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train HGNN for hypergraph representation learning')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to node features file')
    parser.add_argument('--incidence', type=str, required=True,
                        help='Path to hypergraph incidence matrix file')
    parser.add_argument('--output', type=str, default='data/embeddings/',
                        help='Output directory for embeddings')
    parser.add_argument('--dim', type=int, default=500,
                        help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device for computation')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    node_features = np.loadtxt(args.features, dtype=np.float32)
    incidence_matrix = np.loadtxt(args.incidence, dtype=np.float32)

    num_nodes = node_features.shape[0]
    print(f"Data loaded: {num_nodes} nodes, {incidence_matrix.shape[0]} diseases")

    # Initialize model
    model = HGNNModel(
        num_nodes=num_nodes,
        embedding_dim=args.dim,
        device=args.device
    )

    # Train
    print("\nTraining HGNN...")
    node_emb, disease_emb = model.fit(node_features, incidence_matrix)

    # Save
    print("\nSaving embeddings...")
    model.save_embeddings(args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
