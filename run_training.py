# Run CNN Model Training Script
# This script runs the CNN model training using the preprocessed data.

import os
import sys
import numpy as np
import tensorflow as tf
from main_pipeline import DeepfakeDetectionPipeline

# Set up GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✓ GPU detected and configured: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected, using CPU")

def main():
    """Main training function."""
    print("DEEPFAKE AUDIO DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Configuration
    data_dir = "/Users/berkut/Desktop/Projects/deepfakedeneme/model"
    results_dir = "results"
    
    # Training parameters
    EPOCHS = 25  # Reduced for faster training
    BATCH_SIZE = 32
    
    print(f"Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Results directory: {results_dir}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    try:
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = DeepfakeDetectionPipeline(data_dir, results_dir)
        
        # Load preprocessed data
        print("\nLoading preprocessed data...")
        pipeline.load_and_preprocess_data()
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {len(pipeline.train_data['labels'])}")
        print(f"  Validation samples: {len(pipeline.val_data['labels'])}")
        print(f"  Test samples: {len(pipeline.test_data['labels'])}")
        print(f"  Feature dimensions: {pipeline.train_data['features'].shape[1]}")
        print(f"  Spectrogram shape: {pipeline.train_data['spectrograms'].shape[1:]}")
        
        # Train models one by one
        print(f"\n{'='*60}")
        print("TRAINING MODELS")
        print(f"{'='*60}")
        
        # 1. Train Basic CNN
        print("\n1. Training Basic CNN...")
        try:
            basic_model, basic_history = pipeline.train_cnn_model(
                model_type='basic', 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE
            )
            print("Basic CNN training completed successfully!")
        except Exception as e:
            print(f"Basic CNN training failed: {e}")
            return False
        
        # 2. Train Advanced CNN
        print("\n2. Training Advanced CNN...")
        try:
            advanced_model, advanced_history = pipeline.train_cnn_model(
                model_type='advanced', 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE
            )
            print("Advanced CNN training completed successfully!")
        except Exception as e:
            print(f"Advanced CNN training failed: {e}")
            print("Continuing with other models...")
        
        # 3. Train Hybrid Model
        print("\n3. Training Hybrid Model...")
        try:
            hybrid_model, hybrid_history = pipeline.train_hybrid_model(
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE
            )
            print("Hybrid model training completed successfully!")
        except Exception as e:
            print(f"Hybrid model training failed: {e}")
            print("Continuing with evaluation...")
        
        # 4. Evaluate Models
        print(f"\n{'='*60}")
        print("EVALUATING MODELS")
        print(f"{'='*60}")
        
        evaluation_results = {}
        
        # Evaluate each trained model
        for model_name in pipeline.models.keys():
            print(f"\nEvaluating {model_name.upper()}...")
            try:
                results = pipeline.evaluate_model(model_name)
                evaluation_results[model_name] = results
                print(f"✓ {model_name.upper()} evaluation completed!")
                print(f"   Accuracy: {results['accuracy']:.4f}")
                print(f"   F1-Score: {results['f1_score']:.4f}")
                print(f"   ROC-AUC: {results['auc_score']:.4f}")
            except Exception as e:
                print(f"✗ {model_name.upper()} evaluation failed: {e}")
        
        # 5. Generate Report
        if evaluation_results:
            print(f"\n{'='*60}")
            print("GENERATING RESULTS")
            print(f"{'='*60}")
            
            # Find best model
            best_model = max(evaluation_results.keys(), 
                           key=lambda k: evaluation_results[k]['auc_score'])
            best_auc = evaluation_results[best_model]['auc_score']
            
            print(f"\nBest performing model: {best_model.upper()}")
            print(f"Best ROC-AUC: {best_auc:.4f}")
            
            # Analyze misclassified samples
            print(f"\nAnalyzing misclassified samples for best model...")
            try:
                pipeline.analyze_misclassified_samples(best_model, top_n=5)
                print("✓ Misclassification analysis completed!")
            except Exception as e:
                print(f"✗ Misclassification analysis failed: {e}")
            
            # Generate final report
            try:
                pipeline.generate_final_report()
                print("✓ Final report generated!")
            except Exception as e:
                print(f"✗ Report generation failed: {e}")
            
            # Print summary
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print(f"{'='*60}")
            print(f"Models trained: {len(pipeline.models)}")
            print(f"Models evaluated: {len(evaluation_results)}")
            print(f"Results saved in: {results_dir}/")
            
            print(f"\nFinal Results:")
            for model_name, results in evaluation_results.items():
                print(f"  {model_name.upper():<15} AUC: {results['auc_score']:.4f}, "
                      f"Acc: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}")
            
            return True
        else:
            print("\nNo models were successfully evaluated.")
            return False
            
    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    success = main()
    
    if success:
        print(f"\nTraining completed successfully!")
        print("You can find all results, models, and visualizations in the 'results/' directory.")
    else:
        print(f"\nTraining failed. Please check the error messages above.")
