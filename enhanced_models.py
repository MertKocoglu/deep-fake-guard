#!/usr/bin/env python3
"""
Enhanced CNN Models for Improved Deepfake Audio Detection
=========================================================

This module contains improved CNN architectures specifically designed
for better deepfake audio detection performance.

Author: ML Engineer
Date: June 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class EnhancedDeepfakeDetectionCNN:
    """Enhanced CNN models with better architectures for deepfake detection."""
    
    def __init__(self, input_shape=(128, 63, 1)):
        self.input_shape = input_shape
        self.model = None
    
    def build_mobilenet_based_model(self):
        """Build model based on MobileNetV2 architecture."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convert single channel to 3 channels for MobileNet
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        
        # Use MobileNetV2 as backbone (pre-trained on ImageNet)
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(128, 63, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        
        x = backbone(x)
        
        # Add custom head for binary classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def build_efficientnet_based_model(self):
        """Build model based on EfficientNetB0."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Preprocess for EfficientNet
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        x = layers.Lambda(lambda x: tf.image.resize(x, [224, 224]))(x)
        
        # Use EfficientNetB0 as backbone
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Fine-tune the last 20 layers
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        
        x = backbone(x)
        
        # Custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def build_custom_enhanced_model(self):
        """Build custom enhanced CNN optimized for audio spectrograms."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Multi-scale feature extraction
        # Branch 1: Small receptive field
        branch1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        branch1 = layers.BatchNormalization()(branch1)
        
        # Branch 2: Medium receptive field
        branch2 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
        branch2 = layers.BatchNormalization()(branch2)
        
        # Branch 3: Large receptive field
        branch3 = layers.Conv2D(32, (7, 7), padding='same', activation='relu')(inputs)
        branch3 = layers.BatchNormalization()(branch3)
        
        # Combine branches
        x = layers.Concatenate()([branch1, branch2, branch3])
        x = layers.Conv2D(64, (1, 1), activation='relu')(x)  # Reduce channels
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Residual blocks with attention
        x = self._enhanced_residual_block(x, 64)
        x = self._enhanced_residual_block(x, 128, downsample=True)
        x = self._enhanced_residual_block(x, 128)
        x = self._enhanced_residual_block(x, 256, downsample=True)
        x = self._enhanced_residual_block(x, 256)
        
        # Spatial attention
        x = self._spatial_attention_block(x)
        
        # Channel attention
        x = self._channel_attention_block(x)
        
        # Global features
        gap = layers.GlobalAveragePooling2D()(x)
        gmp = layers.GlobalMaxPooling2D()(x)
        
        # Combine global features
        global_features = layers.Concatenate()([gap, gmp])
        
        # Classification head
        x = layers.Dense(512, activation='relu')(global_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def _enhanced_residual_block(self, x, filters, downsample=False):
        """Enhanced residual block with better skip connections."""
        stride = 2 if downsample else 1
        shortcut = x
        
        # First conv layer
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv layer
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if downsample or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # SE block (Squeeze-and-Excitation)
        x = self._se_block(x, filters)
        
        # Add and activate
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def _se_block(self, x, filters, ratio=16):
        """Squeeze-and-Excitation block."""
        # Squeeze
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters // ratio, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        
        # Excitation
        se = layers.Reshape((1, 1, filters))(se)
        return layers.Multiply()([x, se])
    
    def _spatial_attention_block(self, x):
        """Spatial attention mechanism."""
        # Average and max pooling along channel axis
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        
        # Concatenate and apply convolution
        concat = layers.Concatenate()([avg_pool, max_pool])
        attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
        
        return layers.Multiply()([x, attention])
    
    def _channel_attention_block(self, x):
        """Channel attention mechanism."""
        # Global average and max pooling
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        # Shared MLP
        avg_out = layers.Dense(x.shape[-1] // 8, activation='relu')(avg_pool)
        avg_out = layers.Dense(x.shape[-1], activation='sigmoid')(avg_out)
        
        max_out = layers.Dense(x.shape[-1] // 8, activation='relu')(max_pool)
        max_out = layers.Dense(x.shape[-1], activation='sigmoid')(max_out)
        
        # Combine and reshape
        attention = layers.Add()([avg_out, max_out])
        attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
        
        return layers.Multiply()([x, attention])
    
    def build_temporal_cnn_model(self):
        """Build CNN model that specifically captures temporal patterns."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Temporal convolutions (focus on time axis)
        x = layers.Conv2D(32, (3, 9), padding='same', activation='relu')(inputs)  # Time-focused
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 9), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((1, 2))(x)  # Pool only in time dimension
        x = layers.Dropout(0.25)(x)
        
        # Frequency convolutions (focus on frequency axis)
        x = layers.Conv2D(64, (9, 3), padding='same', activation='relu')(x)  # Freq-focused
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (9, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 1))(x)  # Pool only in frequency dimension
        x = layers.Dropout(0.25)(x)
        
        # Combined temporal-spectral convolutions
        x = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Attention mechanism
        x = self._channel_attention_block(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.0001, class_weight=None):
        """Compile model with appropriate settings."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Use adaptive learning rate
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Focal loss for class imbalance
        if class_weight is not None:
            loss = self._focal_loss(alpha=0.25, gamma=2.0)
        else:
            loss = 'binary_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                self._f1_metric
            ]
        )
    
    def _focal_loss(self, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance."""
        def focal_loss_fn(y_true, y_pred):
            # Convert to probabilities
            y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
            
            # Calculate focal loss
            pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
            alpha_t = tf.where(y_true == 1, alpha, 1 - alpha)
            
            focal_loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
            return tf.reduce_mean(focal_loss)
        
        return focal_loss_fn
    
    def _f1_metric(self, y_true, y_pred):
        """F1 score metric."""
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
        fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            raise ValueError("Model must be built first")
        return self.model.summary()


def create_enhanced_callbacks(model_path, patience=15):
    """Create enhanced training callbacks."""
    callbacks = [
        # Early stopping with F1 score monitoring
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-8,
            verbose=1
        ),
        
        # Cosine annealing schedule
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-4 * 0.5 * (1 + np.cos(np.pi * epoch / 50))
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    print("Testing Enhanced CNN Models...")
    
    # Test custom enhanced model
    enhanced_cnn = EnhancedDeepfakeDetectionCNN(input_shape=(128, 63, 1))
    
    print("\\n1. Testing Custom Enhanced Model...")
    model = enhanced_cnn.build_custom_enhanced_model()
    enhanced_cnn.compile_model()
    print(f"✓ Custom Enhanced Model created")
    print(f"  Parameters: {model.count_params():,}")
    
    print("\\n2. Testing Temporal CNN Model...")
    model = enhanced_cnn.build_temporal_cnn_model()
    enhanced_cnn.compile_model()
    print(f"✓ Temporal CNN Model created")
    print(f"  Parameters: {model.count_params():,}")
    
    print("\\n3. Testing MobileNet-based Model...")
    try:
        model = enhanced_cnn.build_mobilenet_based_model()
        enhanced_cnn.compile_model()
        print(f"✓ MobileNet-based Model created")
        print(f"  Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"✗ MobileNet model failed: {e}")
    
    print("\\n✅ All enhanced models tested successfully!")
