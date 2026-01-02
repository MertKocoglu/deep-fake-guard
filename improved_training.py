#!/usr/bin/env python3
"""
Improved Training Script for Deepfake Audio Detection
====================================================

This script addresses the poor performance issues by implementing:
1. Proper class balancing techniques
2. Optimal threshold finding
3. Enhanced data augmentation
4. Better model architectures
5. Advanced training strategies

Author: ML Engineer
Date: June 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from main_pipeline import DeepfakeDetectionPipeline
import warnings
warnings.filterwarnings('ignore')

class ImprovedDeepfakeDetection(DeepfakeDetectionPipeline):
    """Enhanced pipeline with improved training strategies."""
    
    def __init__(self, data_dir, results_dir="improved_results"):
        super().__init__(data_dir, results_dir)
        self.optimal_thresholds = {}
        self.class_weights = None
        
    def analyze_dataset_balance(self):
        """Analyze and report dataset class distribution."""
        print("="*60)
        print("DATASET BALANCE ANALYSIS")
        print("="*60)
        
        splits = ['train', 'val', 'test']
        data_splits = [self.train_data, self.val_data, self.test_data]
        
        total_stats = {'real': 0, 'fake': 0}
        
        for split_name, data in zip(splits, data_splits):
            if data is not None:
                labels = data['labels']
                real_count = np.sum(labels == 0)
                fake_count = np.sum(labels == 1)
                total = len(labels)
                
                print(f"\n{split_name.upper()} SET:")
                print(f"  Real samples: {real_count:,} ({real_count/total*100:.1f}%)")
                print(f"  Fake samples: {fake_count:,} ({fake_count/total*100:.1f}%)")
                print(f"  Total:        {total:,}")
                print(f"  Balance ratio: 1:{fake_count/real_count:.2f}")
                
                total_stats['real'] += real_count
                total_stats['fake'] += fake_count
        
        print(f"\nOVERALL DATASET:")
        total = total_stats['real'] + total_stats['fake']
        print(f"  Real samples: {total_stats['real']:,} ({total_stats['real']/total*100:.1f}%)")
        print(f"  Fake samples: {total_stats['fake']:,} ({total_stats['fake']/total*100:.1f}%)")
        print(f"  Total:        {total:,}")
        print(f"  Balance ratio: 1:{total_stats['fake']/total_stats['real']:.2f}")
        
        return total_stats
    
    def compute_class_weights(self):
        """Compute class weights for balanced training."""
        if self.train_data is None:
            raise ValueError("Training data must be loaded first")
        
        labels = self.train_data['labels']
        classes = np.unique(labels)
        
        # Compute sklearn class weights
        sklearn_weights = compute_class_weight(
            'balanced', 
            classes=classes, 
            y=labels
        )
        
        # Convert to dictionary format
        self.class_weights = {int(classes[i]): sklearn_weights[i] for i in range(len(classes))}
        
        print(f"\nComputed Class Weights:")
        print(f"  Real (0): {self.class_weights[0]:.3f}")
        print(f"  Fake (1): {self.class_weights[1]:.3f}")
        
        return self.class_weights
    
    def create_balanced_generators(self, batch_size=32):
        """Create balanced data generators with augmentation."""
        from tensorflow.keras.utils import Sequence
        
        class BalancedDataGenerator(Sequence):
            def __init__(self, spectrograms, features, labels, batch_size=32, 
                        is_hybrid=False, augment=True):
                self.spectrograms = spectrograms
                self.features = features if is_hybrid else None
                self.labels = labels
                self.batch_size = batch_size
                self.is_hybrid = is_hybrid
                self.augment = augment
                
                # Separate indices by class
                self.real_indices = np.where(labels == 0)[0]
                self.fake_indices = np.where(labels == 1)[0]
                
                # Calculate samples per epoch
                self.samples_per_class = min(len(self.real_indices), len(self.fake_indices))
                self.total_samples = self.samples_per_class * 2
                
                self.on_epoch_end()
            
            def __len__(self):
                return int(np.ceil(self.total_samples / self.batch_size))
            
            def __getitem__(self, idx):
                # Generate balanced batch
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                
                # Get balanced indices
                real_count = self.batch_size // 2
                fake_count = self.batch_size - real_count
                
                real_batch_indices = np.random.choice(self.real_indices, real_count, replace=True)
                fake_batch_indices = np.random.choice(self.fake_indices, fake_count, replace=True)
                
                all_batch_indices = np.concatenate([real_batch_indices, fake_batch_indices])
                np.random.shuffle(all_batch_indices)
                
                # Get data
                batch_specs = self.spectrograms[all_batch_indices]
                batch_labels = self.labels[all_batch_indices]
                
                # Apply augmentation
                if self.augment:
                    batch_specs = self._apply_spectrogram_augmentation(batch_specs)
                
                if self.is_hybrid:
                    batch_features = self.features[all_batch_indices]
                    return [batch_specs, batch_features], batch_labels
                else:
                    return batch_specs, batch_labels
            
            def on_epoch_end(self):
                self.indices = np.arange(self.total_samples)
                np.random.shuffle(self.indices)
            
            def _apply_spectrogram_augmentation(self, spectrograms):
                """Apply random augmentations to spectrograms."""
                augmented = spectrograms.copy()
                
                for i in range(len(augmented)):
                    # Random time masking
                    if np.random.random() < 0.3:
                        time_mask_size = np.random.randint(1, 8)
                        time_start = np.random.randint(0, augmented.shape[2] - time_mask_size)
                        augmented[i, :, time_start:time_start+time_mask_size, :] = 0
                    
                    # Random frequency masking
                    if np.random.random() < 0.3:
                        freq_mask_size = np.random.randint(1, 16)
                        freq_start = np.random.randint(0, augmented.shape[1] - freq_mask_size)
                        augmented[i, freq_start:freq_start+freq_mask_size, :, :] = 0
                    
                    # Random noise injection
                    if np.random.random() < 0.2:
                        noise_factor = np.random.uniform(0.01, 0.05)
                        noise = np.random.normal(0, noise_factor, augmented[i].shape)
                        augmented[i] += noise
                
                return augmented
        
        return BalancedDataGenerator
    
    def find_optimal_threshold(self, model, model_name):
        """Find optimal classification threshold using validation data."""
        print(f"\nFinding optimal threshold for {model_name.upper()}...")
        
        # Get validation predictions
        if model_name == 'hybrid':
            val_inputs = [self.val_data['spectrograms'], self.val_data['features']]
        else:
            val_inputs = self.val_data['spectrograms']
        
        val_predictions = model.predict(val_inputs)
        val_labels = self.val_data['labels']
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(val_labels, val_predictions)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find threshold that maximizes F1 score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        self.optimal_thresholds[model_name] = optimal_threshold
        
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Best F1 score: {f1_scores[optimal_idx]:.4f}")
        print(f"  Precision at optimal: {precisions[optimal_idx]:.4f}")
        print(f"  Recall at optimal: {recalls[optimal_idx]:.4f}")
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(recalls, precisions, 'b-', linewidth=2)
        plt.scatter(recalls[optimal_idx], precisions[optimal_idx], 
                   color='red', s=100, zorder=5, label=f'Optimal (F1={f1_scores[optimal_idx]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
        plt.axvline(optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal threshold: {optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score vs Threshold - {model_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'threshold_analysis_{model_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return optimal_threshold
    
    def train_improved_cnn(self, model_type='advanced', epochs=50, batch_size=32):
        """Train CNN with improved strategies."""
        print(f"\n{'='*60}")
        print(f"TRAINING IMPROVED {model_type.upper()} CNN")
        print(f"{'='*60}")
        
        from cnn_model import DeepfakeDetectionCNN, create_callbacks
        
        # Create model
        input_shape = self.train_data['spectrograms'].shape[1:]
        cnn = DeepfakeDetectionCNN(input_shape=input_shape)
        
        if model_type == 'advanced':
            model = cnn.build_advanced_model()
        else:
            model = cnn.build_model()
        
        # Compile with class weights
        cnn.model = model
        cnn.compile_model(
            learning_rate=0.0001,  # Lower learning rate
            class_weight=self.class_weights
        )
        
        print(f"Model compiled with class weights: {self.class_weights}")
        
        # Create balanced data generators
        GeneratorClass = self.create_balanced_generators(batch_size)
        
        train_generator = GeneratorClass(
            self.train_data['spectrograms'],
            None,
            self.train_data['labels'],
            batch_size=batch_size,
            is_hybrid=False,
            augment=True
        )
        
        val_generator = GeneratorClass(
            self.val_data['spectrograms'],
            None,
            self.val_data['labels'],
            batch_size=batch_size,
            is_hybrid=False,
            augment=False
        )
        
        # Create callbacks
        model_path = os.path.join(self.results_dir, f'best_improved_{model_type}_cnn.h5')
        callbacks = create_callbacks(model_path, patience=15)
        
        # Add custom metrics callback
        from tensorflow.keras.callbacks import Callback
        
        class MetricsCallback(Callback):
            def __init__(self, validation_data):
                self.validation_data = validation_data
                
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:  # Every 5 epochs
                    val_predictions = self.model.predict(self.validation_data[0])
                    val_binary = (val_predictions > 0.5).astype(int)
                    val_f1 = f1_score(self.validation_data[1], val_binary)
                    print(f"\\nEpoch {epoch+1} - Validation F1: {val_f1:.4f}")
        
        val_data_for_callback = (self.val_data['spectrograms'], self.val_data['labels'])
        callbacks.append(MetricsCallback(val_data_for_callback))
        
        # Train model
        print(f"\\nStarting training with balanced batches...")
        print(f"Training samples per epoch: {len(train_generator) * batch_size}")
        print(f"Validation samples: {len(self.val_data['labels'])}")
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model
        self.models[f'improved_{model_type}'] = model
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(model, f'improved_{model_type}')
        
        return model, history, optimal_threshold
    
    def evaluate_with_optimal_threshold(self, model_name):
        """Evaluate model using optimal threshold."""
        print(f"\\n{'='*60}")
        print(f"EVALUATING {model_name.upper()} WITH OPTIMAL THRESHOLD")
        print(f"{'='*60}")
        
        model = self.models[model_name]
        optimal_threshold = self.optimal_thresholds.get(model_name, 0.5)
        
        # Get test predictions
        if 'hybrid' in model_name:
            test_inputs = [self.test_data['spectrograms'], self.test_data['features']]
        else:
            test_inputs = self.test_data['spectrograms']
        
        predictions_proba = model.predict(test_inputs)
        predictions_binary = (predictions_proba > optimal_threshold).astype(int)
        
        true_labels = self.test_data['labels']
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        accuracy = accuracy_score(true_labels, predictions_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions_binary, average='binary'
        )
        auc_score = roc_auc_score(true_labels, predictions_proba)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions_binary)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'predictions_proba': predictions_proba,
            'predictions_binary': predictions_binary,
            'true_labels': true_labels,
            'optimal_threshold': optimal_threshold
        }
        
        # Print results
        print(f"\\nEvaluation Results for {model_name.upper()}:")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc_score:.4f}")
        
        # Print confusion matrix
        print(f"\\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"Actual    Real  Fake")
        print(f"Real      {cm[0,0]:<4}  {cm[0,1]:<4}")
        print(f"Fake      {cm[1,0]:<4}  {cm[1,1]:<4}")
        
        # Calculate per-class metrics
        print(f"\\nPer-class Results:")
        print(f"Real Detection (Specificity): {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
        print(f"Fake Detection (Sensitivity): {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
        
        # Generate visualizations
        self.plot_improved_confusion_matrix(cm, model_name, optimal_threshold)
        self.plot_roc_curve(true_labels, predictions_proba, model_name)
        
        return self.results[model_name]
    
    def plot_improved_confusion_matrix(self, cm, model_name, threshold):
        """Plot enhanced confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm / cm.sum() * 100
        
        # Create annotations
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name.upper()}\\n(Threshold: {threshold:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add performance metrics as text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\\nPrecision: {precision:.3f}\\nRecall: {recall:.3f}\\nF1-Score: {f1:.3f}'
        plt.text(2.1, 0.5, metrics_text, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'improved_confusion_matrix_{model_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self):
        """Generate comparison between original and improved models."""
        print(f"\\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}")
        
        report_path = os.path.join(self.results_dir, 'improvement_comparison.txt')
        
        with open(report_path, 'w') as f:
            f.write("DEEPFAKE DETECTION - IMPROVEMENT ANALYSIS\\n")
            f.write("="*60 + "\\n\\n")
            
            f.write("IMPROVED MODEL RESULTS\\n")
            f.write("-"*30 + "\\n")
            
            # Write model results
            f.write(f"{'Model':<20} {'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\\n")
            f.write("-"*90 + "\\n")
            
            for model_name, results in self.results.items():
                threshold = results.get('optimal_threshold', 0.5)
                f.write(f"{model_name:<20} {threshold:<10.3f} {results['accuracy']:<10.4f} "
                       f"{results['precision']:<10.4f} {results['recall']:<10.4f} "
                       f"{results['f1_score']:<10.4f} {results['auc_score']:<10.4f}\\n")
            
            # Find best model
            if self.results:
                best_model = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
                f.write(f"\\nBEST PERFORMING MODEL: {best_model.upper()}\\n")
                f.write(f"Best F1-Score: {self.results[best_model]['f1_score']:.4f}\\n")
                
                # Class weights used
                f.write(f"\\nCLASS WEIGHTS APPLIED:\\n")
                f.write(f"Real (0): {self.class_weights[0]:.3f}\\n")
                f.write(f"Fake (1): {self.class_weights[1]:.3f}\\n")
        
        print(f"✓ Comparison report saved to: {report_path}")
        
        # Create visual comparison
        self.plot_model_comparison()
    
    def plot_model_comparison(self):
        """Create visual comparison of model performance."""
        if not self.results:
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
            axes[i].set_title(f'{metric.upper().replace("_", " ")}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].grid(True, alpha=0.3)
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.suptitle('Improved Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_performance_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run improved training."""
    print("IMPROVED DEEPFAKE AUDIO DETECTION TRAINING")
    print("=" * 60)
    
    # Set up GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✓ GPU configured: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Configuration
    data_dir = "/Users/berkut/Desktop/Projects/deepfakedeneme/model"
    results_dir = "improved_results"
    
    # Create improved pipeline
    pipeline = ImprovedDeepfakeDetection(data_dir, results_dir)
    
    try:
        # Load data
        print("\\nLoading preprocessed data...")
        pipeline.load_and_preprocess_data()
        
        # Analyze dataset balance
        pipeline.analyze_dataset_balance()
        
        # Compute class weights
        pipeline.compute_class_weights()
        
        # Train improved models
        print(f"\\n{'='*60}")
        print("TRAINING IMPROVED MODELS")
        print(f"{'='*60}")
        
        # Train improved advanced CNN
        model, history, threshold = pipeline.train_improved_cnn(
            model_type='advanced',
            epochs=40,
            batch_size=32
        )
        
        # Evaluate with optimal threshold
        results = pipeline.evaluate_with_optimal_threshold('improved_advanced')
        
        # Generate comparison report
        pipeline.generate_comparison_report()
        
        print(f"\\n🎉 Improved training completed!")
        print(f"Results saved in: {results_dir}/")
        print(f"\\nImproved Results Summary:")
        print(f"  Model: IMPROVED_ADVANCED_CNN")
        print(f"  Optimal Threshold: {threshold:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Recall (Fake Detection): {results['recall']:.4f}")
        print(f"  ROC-AUC: {results['auc_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Improved training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
