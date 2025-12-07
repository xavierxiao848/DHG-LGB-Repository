"""
LightGBM classifier for metabolite-disease association prediction.

This module implements the LightGBM-based binary classifier that operates on
concatenated node-disease embeddings learned from HGNN.
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional
import pickle


class LightGBMClassifier:
    """
    LightGBM classifier for binary metabolite-disease association prediction.

    This classifier takes concatenated embeddings (metabolite + disease) and predicts
    whether the association exists. It uses gradient boosting with L2 regularization
    and supports k-fold cross-validation.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations
    max_depth : int, default=-1
        Maximum tree depth (-1 means no limit)
    learning_rate : float, default=0.1
        Boosting learning rate
    num_leaves : int, default=31
        Maximum number of leaves in one tree
    reg_lambda : float, default=0.1
        L2 regularization term on weights
    random_state : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    models : List[lgb.Booster]
        Trained LightGBM models (one per fold in cross-validation)
    feature_importances : np.ndarray
        Feature importance scores

    Examples
    --------
    >>> # Load embeddings
    >>> node_emb = np.loadtxt('data/embeddings/node_embeddings.txt')
    >>> disease_emb = np.loadtxt('data/embeddings/disease_embeddings.txt')
    >>>
    >>> # Prepare training data
    >>> X_train, y_train = prepare_training_data(associations, node_emb, disease_emb)
    >>>
    >>> # Initialize and train classifier
    >>> clf = LightGBMClassifier(n_estimators=100, reg_lambda=0.1)
    >>> results = clf.fit_with_cv(X_train, y_train, n_folds=5)
    >>>
    >>> # Predict
    >>> y_pred_proba = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        reg_lambda: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1
    ):
        self.params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'reg_alpha': 0.0,  # L1 regularization (disabled)
            'reg_lambda': reg_lambda,  # L2 regularization (enabled)
            'min_data_in_leaf': 20,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'feature_fraction': 1.0,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbosity': verbose
        }

        self.models = []
        self.feature_importances = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None
    ) -> lgb.Booster:
        """
        Train a single LightGBM model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features (shape: [n_samples, n_features])
            Features are concatenated [metabolite_embedding, disease_embedding]
        y_train : np.ndarray
            Training labels (0 or 1)
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
        early_stopping_rounds : int, optional
            Early stopping patience

        Returns
        -------
        lgb.Booster
            Trained model
        """
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)

        # Add validation set if provided
        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Train model
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params['n_estimators'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds else None
        )

        return model

    def fit_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Train with k-fold cross-validation and return comprehensive metrics.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (shape: [n_samples, n_features])
        y : np.ndarray
            Labels (0 or 1)
        n_folds : int, default=5
            Number of folds for cross-validation (paper uses 5-fold)
        shuffle : bool, default=True
            Whether to shuffle data before splitting
        random_state : int, default=42
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary containing:
            - 'y_true_all': concatenated true labels across all folds
            - 'y_pred_proba_all': concatenated predicted probabilities
            - 'y_pred_all': concatenated binary predictions
            - 'fold_results': list of per-fold metrics

        Notes
        -----
        This method performs stratified k-fold cross-validation, training on k-1 folds
        and evaluating on the remaining fold. This ensures all data is used for both
        training and evaluation while maintaining independence.

        The method uses a decision threshold of 0.5 for binary classification, but
        returns probabilities for more flexible threshold selection.
        """
        from ..evaluation.metrics import compute_metrics

        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

        # Storage for results
        y_true_all = []
        y_pred_proba_all = []
        y_pred_all = []
        fold_results = []

        print(f"\nStarting {n_folds}-fold cross-validation...")
        print(f"Total samples: {len(X)} (positive: {np.sum(y)}, negative: {len(y) - np.sum(y)})")
        print(f"Feature dimensionality: {X.shape[1]}")

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"\n--- Fold {fold_idx}/{n_folds} ---")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

            # Train model
            model = self.fit(X_train, y_train)
            self.models.append(model)

            # Predict
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Store predictions
            y_true_all.extend(y_test)
            y_pred_proba_all.extend(y_pred_proba)
            y_pred_all.extend(y_pred)

            # Compute fold metrics
            fold_metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            fold_results.append(fold_metrics)

            # Print fold results
            print(f"Fold {fold_idx} Results:")
            print(f"  AUC: {fold_metrics['auc']:.4f}")
            print(f"  AUPRC: {fold_metrics['auprc']:.4f}")
            print(f"  MCC: {fold_metrics['mcc']:.4f}")
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")

        # Compute overall statistics
        print(f"\n{'='*60}")
        print(f"Cross-Validation Complete")
        print(f"{'='*60}")

        # Compute mean and std across folds
        metric_names = ['accuracy', 'sensitivity', 'specificity', 'precision', 'mcc', 'auc', 'auprc']
        print("\nOverall Performance (Mean ± Std):")
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")

        return {
            'y_true_all': np.array(y_true_all),
            'y_pred_proba_all': np.array(y_pred_proba_all),
            'y_pred_all': np.array(y_pred_all),
            'fold_results': fold_results
        }

    def predict_proba(self, X: np.ndarray, use_best_model: bool = False) -> np.ndarray:
        """
        Predict probabilities for associations.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        use_best_model : bool, default=False
            If True, use the first model only. If False, average predictions
            across all models (ensemble).

        Returns
        -------
        np.ndarray
            Predicted probabilities (values in [0, 1])
        """
        if not self.models:
            raise ValueError("No trained models available. Call fit() or fit_with_cv() first.")

        if use_best_model:
            return self.models[0].predict(X)
        else:
            # Ensemble: average predictions from all folds
            predictions = np.array([model.predict(X) for model in self.models])
            return np.mean(predictions, axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels (0 or 1).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        threshold : float, default=0.5
            Decision threshold

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1)
        """
        y_proba = self.predict_proba(X)
        return (y_proba > threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save all trained models to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Models saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """Load trained models from file."""
        with open(filepath, 'rb') as f:
            self.models = pickle.load(f)
        print(f"Models loaded from: {filepath}")


def prepare_features(
    associations: np.ndarray,
    node_embeddings: np.ndarray,
    disease_embeddings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and labels from associations and embeddings.

    Parameters
    ----------
    associations : np.ndarray
        Array of [metabolite_idx, disease_idx, label] triplets
        Shape: [n_samples, 3]
    node_embeddings : np.ndarray
        Node embeddings from HGNN (shape: [n_nodes, embedding_dim])
    disease_embeddings : np.ndarray
        Disease embeddings from HGNN (shape: [n_diseases, embedding_dim])

    Returns
    -------
    X : np.ndarray
        Feature matrix (shape: [n_samples, 2*embedding_dim])
        Each row is [metabolite_embedding, disease_embedding]
    y : np.ndarray
        Labels (0 or 1)

    Examples
    --------
    >>> # associations: [[metabolite_id, disease_id, label], ...]
    >>> X, y = prepare_features(associations, node_emb, disease_emb)
    >>> print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    """
    X = []
    y = []

    for metabolite_idx, disease_idx, label in associations:
        metabolite_idx = int(metabolite_idx)
        disease_idx = int(disease_idx)

        # Get embeddings
        metabolite_emb = node_embeddings[metabolite_idx]
        disease_emb = disease_embeddings[disease_idx]

        # Concatenate
        feature_vector = np.concatenate([metabolite_emb, disease_emb])

        X.append(feature_vector)
        y.append(label)

    return np.array(X), np.array(y)
