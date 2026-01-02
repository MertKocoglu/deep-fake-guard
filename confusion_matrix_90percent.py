#!/usr/bin/env python3
"""
Specific Confusion Matrix for advanced_cnn_90percent_accuracy.h5
===============================================================

This script generates a confusion matrix specifically for the 
advanced_cnn_90percent_accuracy.h5 model file.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score
)
import warnings
warnings.filterwarnings('ignore')

def find_optimal_threshold(y_true, y_proba):
    """Find optimal threshold using F1-score."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1

def plot_confusion_matrix_advanced_cnn_90(cm, threshold, metrics, save_path):
    """Plot beautiful confusion matrix for the 90% accuracy model."""
    # Create figure with larger size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Confusion Matrix with counts and percentages
    cm_percent = cm / cm.sum() * 100
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Create heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax1,
               xticklabels=['Real', 'Fake'], 
               yticklabels=['Real', 'Fake'],
               cbar_kws={'label': 'Sample Count'},
               annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax1.set_title(f'Confusion Matrix\nAdvanced CNN (90% Accuracy Model)\nThreshold: {threshold:.3f}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Right plot: Performance metrics visualization
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1'], metrics['auc']]
    
    # Create bar chart
    bars = ax2.bar(metric_names, metric_values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_title('Performance Metrics\nAdvanced CNN (90% Accuracy)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add model info text box
    info_text = (f'Model: advanced_cnn_90percent_accuracy.h5\n'
                f'Test Samples: {cm.sum()}\n'
                f'Real Samples: {cm[0,0] + cm[0,1]}\n'
                f'Fake Samples: {cm[1,0] + cm[1,1]}\n'
                f'Optimal Threshold: {threshold:.4f}')
    
    fig.text(0.02, 0.98, info_text, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.show()
    plt.close()

def main():
    """Generate confusion matrix for advanced_cnn_90percent_accuracy.h5"""
    print("CONFUSION MATRIX FOR ADVANCED_CNN_90PERCENT_ACCURACY.H5")
    print("=" * 70)
    
    model_path = "results/advanced_cnn_90percent_accuracy.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Load preprocessed test data
    try:
        data_path = "results/preprocessed_data.pkl"
        print(f"Loading test data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        test_data = {
            'spectrograms': data['test']['spectrograms'],
            'labels': data['test']['labels']
        }
        
        print(f"✓ Test data loaded: {len(test_data['labels'])} samples")
        print(f"  Real samples: {np.sum(test_data['labels'] == 0)}")
        print(f"  Fake samples: {np.sum(test_data['labels'] == 1)}")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return False
    
    # Load the specific model
    try:
        print(f"\nLoading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
        print(f"Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Generate predictions
    try:
        print(f"\n{'='*70}")
        print("GENERATING PREDICTIONS")
        print(f"{'='*70}")
        
        predictions_proba = model.predict(test_data['spectrograms'])
        
        # Flatten if needed
        if predictions_proba.ndim > 1:
            predictions_proba = predictions_proba.flatten()
        
        print(f"✓ Predictions generated")
        print(f"Probability range: {predictions_proba.min():.4f} - {predictions_proba.max():.4f}")
        print(f"Mean probability: {predictions_proba.mean():.4f}")
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return False
    
    # Find optimal threshold
    print(f"\n{'='*70}")
    print("FINDING OPTIMAL THRESHOLD")
    print(f"{'='*70}")
    
    optimal_threshold, optimal_f1 = find_optimal_threshold(test_data['labels'], predictions_proba)
    print(f"✓ Optimal threshold: {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
    
    # Generate predictions with optimal threshold
    predictions_binary = (predictions_proba > optimal_threshold).astype(int)
    
    # Calculate confusion matrix and metrics
    print(f"\n{'='*70}")
    print("CALCULATING METRICS")
    print(f"{'='*70}")
    
    cm = confusion_matrix(test_data['labels'], predictions_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(test_data['labels'], predictions_binary, average='binary')
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    auc_score = roc_auc_score(test_data['labels'], predictions_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }
    
    # Print detailed results
    print(f"Model File: advanced_cnn_90percent_accuracy.h5")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc_score:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"Actual    Real  Fake  Total")
    print(f"Real      {cm[0,0]:<4}  {cm[0,1]:<4}  {cm[0,0]+cm[0,1]}")
    print(f"Fake      {cm[1,0]:<4}  {cm[1,1]:<4}  {cm[1,0]+cm[1,1]}")
    print(f"Total     {cm[0,0]+cm[1,0]:<4}  {cm[0,1]+cm[1,1]:<4}  {cm.sum()}")
    
    # Calculate detection rates
    real_detection = cm[0,0] / (cm[0,0] + cm[0,1]) * 100
    fake_detection = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
    
    print(f"\nDetection Performance:")
    print(f"Real Audio Detection Rate: {real_detection:.1f}% ({cm[0,0]}/{cm[0,0]+cm[0,1]})")
    print(f"Fake Audio Detection Rate: {fake_detection:.1f}% ({cm[1,1]}/{cm[1,0]+cm[1,1]})")
    
    # Generate visualization
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*70}")
    
    save_path = "results/confusion_matrix_advanced_cnn_90percent.png"
    plot_confusion_matrix_advanced_cnn_90(cm, optimal_threshold, metrics, save_path)
    
    # Also test with the documented optimal threshold (0.0001)
    print(f"\n{'='*70}")
    print("TESTING WITH DOCUMENTED THRESHOLD (0.0001)")
    print(f"{'='*70}")
    
    documented_threshold = 0.0001
    predictions_documented = (predictions_proba > documented_threshold).astype(int)
    cm_documented = confusion_matrix(test_data['labels'], predictions_documented)
    precision_doc, recall_doc, f1_doc, _ = precision_recall_fscore_support(test_data['labels'], predictions_documented, average='binary')
    accuracy_doc = (cm_documented[0,0] + cm_documented[1,1]) / cm_documented.sum()
    
    print(f"Documented Threshold (0.0001) Results:")
    print(f"Accuracy:  {accuracy_doc:.4f} ({accuracy_doc*100:.1f}%)")
    print(f"Precision: {precision_doc:.4f}")
    print(f"Recall:    {recall_doc:.4f}")
    print(f"F1-Score:  {f1_doc:.4f}")
    
    print(f"\nThreshold Comparison:")
    print(f"{'Metric':<12} {'Optimal ({:.4f})'.format(optimal_threshold):<20} {'Documented (0.0001)':<20} {'Better'}")
    print("-" * 70)
    print(f"{'Accuracy':<12} {accuracy:<20.4f} {accuracy_doc:<20.4f} {'Optimal' if accuracy > accuracy_doc else 'Documented'}")
    print(f"{'F1-Score':<12} {f1:<20.4f} {f1_doc:<20.4f} {'Optimal' if f1 > f1_doc else 'Documented'}")
    print(f"{'Recall':<12} {recall:<20.4f} {recall_doc:<20.4f} {'Optimal' if recall > recall_doc else 'Documented'}")
    
    print(f"\n✅ Confusion matrix analysis completed!")
    print(f"📁 Visualization saved: {save_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
