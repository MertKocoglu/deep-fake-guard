#!/usr/bin/env python3
"""
CNN Model for Deepfake Audio Detection
======================================

This module defines the Convolutional Neural Network architecture
for classifying audio as AI-generated (fake) or human (real).

Author: ML Engineer
Date: June 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class DeepfakeDetectionCNN:
    """CNN model for deepfake audio detection."""
    
    def __init__(self, input_shape=(128, 63, 1), num_classes=2):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of mel-spectrogram input (height, width, channels)
            num_classes: Number of classes (2 for binary classification)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build the CNN architecture."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling instead of flatten to reduce parameters
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model = model
        return model
    
    def build_advanced_model(self):
        """Build an advanced CNN with residual connections."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(32, (7, 7), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64, stride=2)
        x = self._residual_block(x, 64, stride=1)
        
        x = self._residual_block(x, 128, stride=2)
        x = self._residual_block(x, 128, stride=1)
        
        x = self._residual_block(x, 256, stride=2)
        x = self._residual_block(x, 256, stride=1)
        
        # Attention mechanism
        x = self._attention_block(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def _residual_block(self, x, filters, stride=1):
        """Create a residual block."""
        shortcut = x
        
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    def _attention_block(self, x):
        """Add attention mechanism."""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        avg_out = layers.Dense(x.shape[-1] // 8, activation='relu')(avg_pool)
        avg_out = layers.Dense(x.shape[-1], activation='sigmoid')(avg_out)
        
        max_out = layers.Dense(x.shape[-1] // 8, activation='relu')(max_pool)
        max_out = layers.Dense(x.shape[-1], activation='sigmoid')(max_out)
        
        channel_attention = layers.Add()([avg_out, max_out])
        channel_attention = layers.Reshape((1, 1, x.shape[-1]))(channel_attention)
        
        x = layers.Multiply()([x, channel_attention])
        return x
    
    def compile_model(self, learning_rate=0.001, class_weight=None):
        """Compile the model with appropriate loss and metrics."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Use weighted loss if class imbalance exists
        if class_weight is not None:
            loss = self._weighted_binary_crossentropy(class_weight)
        else:
            loss = 'binary_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
    
    def _weighted_binary_crossentropy(self, class_weight):
        """Create weighted binary crossentropy loss."""
        def weighted_loss(y_true, y_pred):
            # Apply class weights
            weight_0 = class_weight[0]  # Weight for class 0 (real)
            weight_1 = class_weight[1]  # Weight for class 1 (fake)
            
            # Calculate weighted loss
            loss_0 = weight_0 * y_true * tf.math.log(y_pred + 1e-7)
            loss_1 = weight_1 * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
            
            return -tf.reduce_mean(loss_0 + loss_1)
        
        return weighted_loss
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            raise ValueError("Model must be built first")
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model must be built first")
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
        return self.model

class HybridModel:
    """Hybrid model combining CNN for spectrograms and MLP for handcrafted features."""
    
    def __init__(self, spectrogram_shape=(128, 63, 1), features_dim=50):
        self.spectrogram_shape = spectrogram_shape
        self.features_dim = features_dim
        self.model = None
    
    def build_model(self):
        """Build hybrid model with two input branches."""
        # Spectrogram branch (CNN)
        spec_input = layers.Input(shape=self.spectrogram_shape, name='spectrogram')
        
        x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spec_input)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(256, activation='relu')(x1)
        x1 = layers.Dropout(0.5)(x1)
        
        # Features branch (MLP)
        features_input = layers.Input(shape=(self.features_dim,), name='features')
        
        x2 = layers.Dense(128, activation='relu')(features_input)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        x2 = layers.Dense(64, activation='relu')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        # Combine branches
        combined = layers.Concatenate()([x1, x2])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.5)(combined)
        
        outputs = layers.Dense(1, activation='sigmoid')(combined)
        
        self.model = keras.Model(inputs=[spec_input, features_input], outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the hybrid model."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

def create_callbacks(model_path, patience=10):
    """Create training callbacks."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

if __name__ == "__main__":
    # Example usage
    print("Testing CNN Model Architecture...")
    
    # Test basic CNN
    cnn = DeepfakeDetectionCNN(input_shape=(128, 63, 1))
    model = cnn.build_model()
    print("\nBasic CNN Architecture:")
    print(model.summary())
    
    # Test advanced CNN
    print("\n" + "="*50)
    print("Testing Advanced CNN Architecture...")
    cnn_advanced = DeepfakeDetectionCNN(input_shape=(128, 63, 1))
    model_advanced = cnn_advanced.build_advanced_model()
    print("\nAdvanced CNN Architecture:")
    print(model_advanced.summary())
    
    # Test hybrid model
    print("\n" + "="*50)
    print("Testing Hybrid Model Architecture...")
    hybrid = HybridModel(spectrogram_shape=(128, 63, 1), features_dim=50)
    model_hybrid = hybrid.build_model()
    print("\nHybrid Model Architecture:")
    print(model_hybrid.summary())
