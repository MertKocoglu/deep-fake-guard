#!/usr/bin/env python3
"""
Main Training and Evaluation Script for Deepfake Audio Detection
================================================================

This script orchestrates the complete pipeline:
1. Data preprocessing and feature extraction
2. Model training with multiple architectures
3. Comprehensive evaluation and reporting
4. Analysis of misclassified samples

Author: ML Engineer
Date: June 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings('ignore')

from audio_preprocessing import AudioFeatureExtractor, DatasetProcessor
from cnn_model import DeepfakeDetectionCNN, HybridModel, create_callbacks

class DeepfakeDetectionPipeline:
    """Complete pipeline for deepfake audio detection."""
    
    def __init__(self, data_dir, results_dir="results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        self.processor = DatasetProcessor(data_dir, self.feature_extractor)
        self.scaler = StandardScaler()
        
        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Models
        self.models = {}
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess all dataset splits."""
        print("="*50)
        print("LOADING AND PREPROCESSING DATA")
        print("="*50)
        
        # Check if preprocessed data exists
        data_path = os.path.join(self.results_dir, "preprocessed_data.pkl")
        scaler_path = os.path.join(self.results_dir, "feature_scaler.pkl")
        
        if os.path.exists(data_path) and os.path.exists(scaler_path):
            print("Found existing preprocessed data. Loading...")
            self.load_preprocessed_data()
        else:
            print("No preprocessed data found. Processing from scratch...")
            
            # Process training data
            print("\n1. Processing training data...")
            train_features, train_mel_specs, train_labels, train_paths = \
                self.processor.process_dataset('training')
            
            # Process validation data
            print("\n2. Processing validation data...")
            val_features, val_mel_specs, val_labels, val_paths = \
                self.processor.process_dataset('validation')
            
            # Process test data
            print("\n3. Processing test data...")
            test_features, test_mel_specs, test_labels, test_paths = \
                self.processor.process_dataset('testing')
            
            # Normalize features
            print("\n4. Normalizing features...")
            train_features_scaled = self.scaler.fit_transform(train_features)
            val_features_scaled = self.scaler.transform(val_features)
            test_features_scaled = self.scaler.transform(test_features)
            
            # Prepare mel-spectrograms for CNN
            train_mel_specs = self.prepare_spectrograms(train_mel_specs)
            val_mel_specs = self.prepare_spectrograms(val_mel_specs)
            test_mel_specs = self.prepare_spectrograms(test_mel_specs)
            
            # Store data
            self.train_data = {
                'features': train_features_scaled,
                'spectrograms': train_mel_specs,
                'labels': train_labels,
                'paths': train_paths
            }
            
            self.val_data = {
                'features': val_features_scaled,
                'spectrograms': val_mel_specs,
                'labels': val_labels,
                'paths': val_paths
            }
            
            self.test_data = {
                'features': test_features_scaled,
                'spectrograms': test_mel_specs,
                'labels': test_labels,
                'paths': test_paths
            }
            
            # Save preprocessed data
            self.save_preprocessed_data()
        
        print(f"\nData preprocessing completed!")
        print(f"Training samples: {len(self.train_data['labels'])}")
        print(f"Validation samples: {len(self.val_data['labels'])}")
        print(f"Test samples: {len(self.test_data['labels'])}")
        
        # Print class distribution
        self.print_class_distribution()
    
    def prepare_spectrograms(self, spectrograms):
        """Prepare spectrograms for CNN input."""
        # Add channel dimension
        spectrograms = spectrograms[..., np.newaxis]
        
        # Normalize to [0, 1]
        spectrograms = (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min())
        
        return spectrograms
    
    def print_class_distribution(self):
        """Print class distribution for all splits."""
        print("\nClass Distribution:")
        print("-" * 30)
        
        for split_name, data in [("Training", self.train_data), 
                                ("Validation", self.val_data), 
                                ("Test", self.test_data)]:
            labels = data['labels']
            real_count = np.sum(labels == 0)
            fake_count = np.sum(labels == 1)
            total = len(labels)
            
            print(f"{split_name}:")
            print(f"  Real: {real_count} ({real_count/total*100:.1f}%)")
            print(f"  Fake: {fake_count} ({fake_count/total*100:.1f}%)")
            print(f"  Total: {total}")
    
    def save_preprocessed_data(self):
        """Save preprocessed data for future use."""
        data_path = os.path.join(self.results_dir, "preprocessed_data.pkl")
        scaler_path = os.path.join(self.results_dir, "feature_scaler.pkl")
        
        with open(data_path, 'wb') as f:
            pickle.dump({
                'train': self.train_data,
                'val': self.val_data,
                'test': self.test_data
            }, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Preprocessed data saved to {data_path}")
    
    def load_preprocessed_data(self):
        """Load previously preprocessed data."""
        data_path = os.path.join(self.results_dir, "preprocessed_data.pkl")
        scaler_path = os.path.join(self.results_dir, "feature_scaler.pkl")
        
        print("Loading preprocessed data...")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.train_data = data['train']
            self.val_data = data['val']
            self.test_data = data['test']
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("✓ Preprocessed data loaded successfully!")
    
    def train_cnn_model(self, model_type='basic', epochs=50, batch_size=32):
        """Train CNN model on mel-spectrograms."""
        print(f"\n{'='*50}")
        print(f"TRAINING {model_type.upper()} CNN MODEL")
        print(f"{'='*50}")
        
        # Initialize model
        input_shape = self.train_data['spectrograms'].shape[1:]
        cnn = DeepfakeDetectionCNN(input_shape=input_shape)
        
        if model_type == 'basic':
            model = cnn.build_model()
        else:
            model = cnn.build_advanced_model()
        
        # Compile model
        cnn.compile_model(learning_rate=0.001)
        
        # Get the compiled model from cnn object
        model = cnn.model
        
        print(f"\nModel architecture:")
        print(model.summary())
        
        # Prepare callbacks
        model_path = os.path.join(self.results_dir, f"best_{model_type}_cnn_model.h5")
        callbacks = create_callbacks(model_path, patience=15)
        
        # Train model
        print(f"\nStarting training...")
        history = model.fit(
            self.train_data['spectrograms'],
            self.train_data['labels'],
            validation_data=(self.val_data['spectrograms'], self.val_data['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model and history
        self.models[f'{model_type}_cnn'] = model
        self.save_training_history(history, f'{model_type}_cnn')
        
        print(f"\n{model_type.upper()} CNN training completed!")
        return model, history
    
    def train_hybrid_model(self, epochs=50, batch_size=32):
        """Train hybrid model combining CNN and MLP."""
        print(f"\n{'='*50}")
        print(f"TRAINING HYBRID MODEL")
        print(f"{'='*50}")
        
        # Initialize hybrid model
        spec_shape = self.train_data['spectrograms'].shape[1:]
        features_dim = self.train_data['features'].shape[1]
        
        hybrid = HybridModel(spectrogram_shape=spec_shape, features_dim=features_dim)
        model = hybrid.build_model()
        hybrid.compile_model(learning_rate=0.001)
        
        # Get the compiled model from hybrid object
        model = hybrid.model
        
        print(f"\nHybrid model architecture:")
        print(model.summary())
        
        # Prepare data
        train_inputs = [self.train_data['spectrograms'], self.train_data['features']]
        val_inputs = [self.val_data['spectrograms'], self.val_data['features']]
        
        # Prepare callbacks
        model_path = os.path.join(self.results_dir, "best_hybrid_model.h5")
        callbacks = create_callbacks(model_path, patience=15)
        
        # Train model
        print(f"\nStarting training...")
        history = model.fit(
            train_inputs,
            self.train_data['labels'],
            validation_data=(val_inputs, self.val_data['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model and history
        self.models['hybrid'] = model
        self.save_training_history(history, 'hybrid')
        
        print(f"\nHybrid model training completed!")
        return model, history
    
    def evaluate_model(self, model_name, model=None):
        """Comprehensive model evaluation."""
        print(f"\n{'='*50}")
        print(f"EVALUATING {model_name.upper()} MODEL")
        print(f"{'='*50}")
        
        if model is None:
            model = self.models[model_name]
        
        # Prepare test data
        if model_name == 'hybrid':
            test_inputs = [self.test_data['spectrograms'], self.test_data['features']]
        else:
            test_inputs = self.test_data['spectrograms']
        
        # Get predictions
        predictions_proba = model.predict(test_inputs)
        predictions_binary = (predictions_proba > 0.5).astype(int)
        
        true_labels = self.test_data['labels']
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions_binary, average='binary'
        )
        auc_score = roc_auc_score(true_labels, predictions_proba)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions_binary)
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions_binary,
            target_names=['Real', 'Fake'],
            output_dict=True
        )
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions_proba': predictions_proba,
            'predictions_binary': predictions_binary,
            'true_labels': true_labels
        }
        
        # Print results
        print(f"\nEvaluation Results for {model_name.upper()}:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc_score:.4f}")
        
        # Generate visualizations
        self.plot_confusion_matrix(cm, model_name)
        self.plot_roc_curve(true_labels, predictions_proba, model_name)
        
        return self.results[model_name]
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'confusion_matrix_{model_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, true_labels, predictions_proba, model_name):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(true_labels, predictions_proba)
        auc_score = roc_auc_score(true_labels, predictions_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name.upper()} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'roc_curve_{model_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_history(self, history, model_name):
        """Save and plot training history."""
        # Save history
        history_path = os.path.join(self.results_dir, f'training_history_{model_name}.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        
        # Plot training curves
        self.plot_training_curves(history.history, model_name)
    
    def plot_training_curves(self, history, model_name):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(history['auc'], label='Training AUC')
        axes[1, 0].plot(history['val_auc'], label='Validation AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision and Recall
        axes[1, 1].plot(history['precision'], label='Training Precision')
        axes[1, 1].plot(history['val_precision'], label='Validation Precision')
        axes[1, 1].plot(history['recall'], label='Training Recall')
        axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {model_name.upper()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'training_curves_{model_name}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_misclassified_samples(self, model_name, top_n=5):
        """Analyze misclassified samples."""
        print(f"\n{'='*50}")
        print(f"ANALYZING MISCLASSIFIED SAMPLES - {model_name.upper()}")
        print(f"{'='*50}")
        
        results = self.results[model_name]
        true_labels = results['true_labels']
        predictions_binary = results['predictions_binary']
        predictions_proba = results['predictions_proba'].flatten()
        
        # Find misclassified samples
        misclassified_mask = (true_labels != predictions_binary.flatten())
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Get probabilities for misclassified samples
        misclassified_probs = predictions_proba[misclassified_indices]
        misclassified_true = true_labels[misclassified_indices]
        misclassified_pred = predictions_binary.flatten()[misclassified_indices]
        
        print(f"\nTotal misclassified samples: {len(misclassified_indices)}")
        
        # Find most uncertain predictions (closest to 0.5)
        uncertainty = np.abs(misclassified_probs - 0.5)
        most_uncertain_indices = misclassified_indices[np.argsort(uncertainty)[:top_n]]
        
        print(f"\nTop {top_n} most uncertain misclassified samples:")
        print("-" * 60)
        
        analysis_results = []
        
        for i, idx in enumerate(most_uncertain_indices):
            true_label = "Real" if true_labels[idx] == 0 else "Fake"
            pred_label = "Real" if predictions_binary[idx] == 0 else "Fake"
            prob = predictions_proba[idx]
            file_path = self.test_data['paths'][idx]
            filename = os.path.basename(file_path)
            
            print(f"\n{i+1}. File: {filename}")
            print(f"   True Label: {true_label}")
            print(f"   Predicted: {pred_label}")
            print(f"   Probability (Fake): {prob:.4f}")
            print(f"   Uncertainty: {abs(prob - 0.5):.4f}")
            
            analysis_results.append({
                'filename': filename,
                'file_path': file_path,
                'true_label': true_label,
                'predicted_label': pred_label,
                'probability': prob,
                'uncertainty': abs(prob - 0.5)
            })
        
        # Save analysis results
        analysis_df = pd.DataFrame(analysis_results)
        analysis_path = os.path.join(self.results_dir, f'misclassified_analysis_{model_name}.csv')
        analysis_df.to_csv(analysis_path, index=False)
        
        return analysis_results
    
    def generate_final_report(self):
        """Generate comprehensive evaluation report."""
        print(f"\n{'='*50}")
        print("GENERATING FINAL EVALUATION REPORT")
        print(f"{'='*50}")
        
        report_path = os.path.join(self.results_dir, "evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("DEEPFAKE AUDIO DETECTION - EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training samples: {len(self.train_data['labels'])}\n")
            f.write(f"Validation samples: {len(self.val_data['labels'])}\n")
            f.write(f"Test samples: {len(self.test_data['labels'])}\n\n")
            
            # Model comparison
            f.write("MODEL COMPARISON\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n")
            f.write("-" * 70 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"{model_name:<15} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                       f"{results['recall']:<10.4f} {results['f1_score']:<10.4f} {results['auc_score']:<10.4f}\n")
            
            f.write(f"\n")
            
            # Detailed results for each model
            for model_name, results in self.results.items():
                f.write(f"\nDETAILED RESULTS - {model_name.upper()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"F1-Score: {results['f1_score']:.4f}\n")
                f.write(f"ROC-AUC: {results['auc_score']:.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                cm = results['confusion_matrix']
                f.write(f"                 Predicted\n")
                f.write(f"Actual    Real  Fake\n")
                f.write(f"Real      {cm[0,0]:<4}  {cm[0,1]:<4}\n")
                f.write(f"Fake      {cm[1,0]:<4}  {cm[1,1]:<4}\n\n")
        
        print(f"Evaluation report saved to {report_path}")
    
    def run_full_pipeline(self, epochs=50, batch_size=32):
        """Run the complete detection pipeline."""
        print("STARTING DEEPFAKE AUDIO DETECTION PIPELINE")
        print("=" * 60)
        
        # 1. Load and preprocess data
        self.load_and_preprocess_data()
        
        # 2. Train models
        print(f"\n{'='*60}")
        print("TRAINING MODELS")
        print(f"{'='*60}")
        
        # Train basic CNN
        basic_cnn, basic_history = self.train_cnn_model('basic', epochs, batch_size)
        
        # Train advanced CNN
        advanced_cnn, advanced_history = self.train_cnn_model('advanced', epochs, batch_size)
        
        # Train hybrid model
        hybrid_model, hybrid_history = self.train_hybrid_model(epochs, batch_size)
        
        # 3. Evaluate all models
        print(f"\n{'='*60}")
        print("EVALUATING MODELS")
        print(f"{'='*60}")
        
        self.evaluate_model('basic_cnn')
        self.evaluate_model('advanced_cnn')
        self.evaluate_model('hybrid')
        
        # 4. Analyze misclassified samples for best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['auc_score'])
        print(f"\nBest performing model: {best_model} (AUC: {self.results[best_model]['auc_score']:.4f})")
        
        self.analyze_misclassified_samples(best_model)
        
        # 5. Generate final report
        self.generate_final_report()
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved in: {self.results_dir}")

if __name__ == "__main__":
    # Initialize and run pipeline
    data_dir = "/Users/berkut/Desktop/Projects/deepfakedeneme/model"
    
    pipeline = DeepfakeDetectionPipeline(data_dir)
    pipeline.run_full_pipeline(epochs=30, batch_size=32)  # Reduced epochs for faster execution
