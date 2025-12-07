"""
Train LightGBM classifier with 5-fold cross-validation.

This script performs the complete training and evaluation pipeline:
1. Load embeddings from HGNN
2. Prepare training data (positive + negative samples)
3. Train LightGBM with 5-fold cross-validation
4. Compute comprehensive metrics
5. Save results and trained models
"""

import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import LightGBMClassifier, prepare_features
from src.evaluation.metrics import print_metrics_summary, plot_roc_curve, plot_pr_curve
from src.utils.logger import setup_logger
from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM classifier')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--node-emb', type=str, default='data/embeddings/node_embeddings.txt',
                        help='Path to node embeddings')
    parser.add_argument('--disease-emb', type=str, default='data/embeddings/disease_embeddings.txt',
                        help='Path to disease embeddings')
    parser.add_argument('--samples', type=str, default='data/processed/samples.txt',
                        help='Path to samples file (metabolite disease label)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger('Train-Classifier', log_dir=os.path.join(args.output_dir, 'logs'))

    logger.info("="*70)
    logger.info("DHG-LGB Classifier Training")
    logger.info("="*70)

    # Load embeddings
    logger.info("\n1. Loading embeddings...")
    node_embeddings = np.loadtxt(args.node_emb, dtype=np.float32)
    disease_embeddings = np.loadtxt(args.disease_emb, dtype=np.float32)
    logger.info(f"   Node embeddings: {node_embeddings.shape}")
    logger.info(f"   Disease embeddings: {disease_embeddings.shape}")

    # Load samples
    logger.info("\n2. Loading samples...")
    samples = np.loadtxt(args.samples, dtype=np.int32)
    logger.info(f"   Total samples: {len(samples)}")
    logger.info(f"   Positive: {np.sum(samples[:, 2])}")
    logger.info(f"   Negative: {len(samples) - np.sum(samples[:, 2])}")

    # Prepare features
    logger.info("\n3. Preparing features...")
    X, y = prepare_features(samples, node_embeddings, disease_embeddings)
    logger.info(f"   Feature matrix: {X.shape}")
    logger.info(f"   Labels: {y.shape}")

    # Initialize classifier
    logger.info("\n4. Initializing LightGBM classifier...")
    clf_params = config['classifier']
    clf = LightGBMClassifier(
        n_estimators=clf_params['n_estimators'],
        max_depth=clf_params['max_depth'],
        learning_rate=clf_params['learning_rate'],
        num_leaves=clf_params['num_leaves'],
        reg_lambda=clf_params['reg_lambda'],
        random_state=clf_params['random_state']
    )
    logger.info(f"   Parameters: {clf.params}")

    # Cross-validation
    logger.info("\n5. Training with 5-fold cross-validation...")
    cv_config = config['cross_validation']
    results = clf.fit_with_cv(
        X, y,
        n_folds=cv_config['n_folds'],
        shuffle=cv_config['shuffle'],
        random_state=cv_config['random_state']
    )

    # Print summary
    print_metrics_summary(results['fold_results'])

    # Save results
    logger.info("\n6. Saving results...")
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)

    # Save metrics
    np.savetxt(
        os.path.join(args.output_dir, 'metrics', 'predictions.txt'),
        np.column_stack([results['y_true_all'], results['y_pred_proba_all'], results['y_pred_all']]),
        header='y_true y_pred_proba y_pred',
        fmt='%d %.6f %d'
    )

    # Plot curves
    plot_roc_curve(
        results['y_true_all'],
        results['y_pred_proba_all'],
        save_path=os.path.join(args.output_dir, 'figures', 'roc_curve.png')
    )
    plot_pr_curve(
        results['y_true_all'],
        results['y_pred_proba_all'],
        save_path=os.path.join(args.output_dir, 'figures', 'pr_curve.png')
    )

    # Save models
    clf.save(os.path.join(args.output_dir, 'models', 'lightgbm_models.pkl'))

    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
