#!/usr/bin/env python3
"""
Enhanced Audio Preprocessing for Better Deepfake Detection
==========================================================

This module provides enhanced preprocessing techniques specifically
designed to improve deepfake audio detection performance.

Author: ML Engineer
Date: June 2025
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

class EnhancedAudioFeatureExtractor:
    """Enhanced feature extractor with advanced techniques for deepfake detection."""
    
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=13, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.scaler = RobustScaler()  # More robust to outliers
    
    def load_audio_enhanced(self, file_path, duration=3.0, augment=False):
        """Enhanced audio loading with optional augmentation."""
        try:
            # Load audio with higher quality
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration, mono=True)
            
            # Ensure exact duration
            target_length = int(self.sample_rate * duration)
            if len(audio) < target_length:
                # Pad with reflection instead of zeros
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='reflect')
            elif len(audio) > target_length:
                # Take center portion instead of beginning
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Apply augmentation if requested
            if augment:
                audio = self._apply_audio_augmentation(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _apply_audio_augmentation(self, audio):
        """Apply various audio augmentations."""
        augmented = audio.copy()
        
        # Time stretching (slight speed change)
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
            # Ensure same length
            if len(augmented) != len(audio):
                if len(augmented) > len(audio):
                    augmented = augmented[:len(audio)]
                else:
                    augmented = np.pad(augmented, (0, len(audio) - len(augmented)), mode='reflect')
        
        # Pitch shifting
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(augmented, sr=self.sample_rate, n_steps=n_steps)
        
        # Add noise
        if np.random.random() < 0.3:
            noise_factor = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_factor, len(augmented))
            augmented = augmented + noise
        
        # Dynamic range compression
        if np.random.random() < 0.2:
            augmented = np.sign(augmented) * np.power(np.abs(augmented), 0.8)
        
        # Normalize after augmentation
        if np.max(np.abs(augmented)) > 0:
            augmented = augmented / np.max(np.abs(augmented))
        
        return augmented
    
    def extract_enhanced_mel_spectrogram(self, audio):
        """Extract enhanced mel-spectrogram with multiple representations."""
        # Standard mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=2048,
            fmin=50,  # Focus on relevant frequency range
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Delta features (temporal derivatives)
        mel_delta = librosa.feature.delta(mel_spec_db)
        mel_delta2 = librosa.feature.delta(mel_spec_db, order=2)
        
        # Stack features
        enhanced_mel = np.stack([mel_spec_db, mel_delta, mel_delta2], axis=-1)
        
        return enhanced_mel
    
    def extract_advanced_spectral_features(self, audio):
        """Extract advanced spectral features."""
        features = {}
        
        # Standard spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Statistical features
        for name, feature in [
            ('spectral_centroid', spectral_centroids),
            ('spectral_rolloff', spectral_rolloff),
            ('spectral_bandwidth', spectral_bandwidth),
            ('zcr', zero_crossing_rate)
        ]:
            features[f'{name}_mean'] = np.mean(feature)
            features[f'{name}_std'] = np.std(feature)
            features[f'{name}_median'] = np.median(feature)
            features[f'{name}_skew'] = skew(feature)
            features[f'{name}_kurtosis'] = kurtosis(feature)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spectral_contrast[i])
        
        # Tonnetz (harmonic features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])
        
        return features
    
    def extract_prosodic_features(self, audio):
        """Extract prosodic features (rhythm, stress, intonation)."""
        features = {}
        
        # Fundamental frequency (F0)
        f0 = librosa.yin(audio, fmin=50, fmax=400)
        valid_f0 = f0[f0 > 0]  # Remove unvoiced segments
        
        if len(valid_f0) > 0:
            features['f0_mean'] = np.mean(valid_f0)
            features['f0_std'] = np.std(valid_f0)
            features['f0_median'] = np.median(valid_f0)
            features['f0_range'] = np.max(valid_f0) - np.min(valid_f0)
            features['f0_q25'] = np.percentile(valid_f0, 25)
            features['f0_q75'] = np.percentile(valid_f0, 75)
            features['voicing_rate'] = len(valid_f0) / len(f0)
        else:
            for key in ['f0_mean', 'f0_std', 'f0_median', 'f0_range', 'f0_q25', 'f0_q75', 'voicing_rate']:
                features[key] = 0
        
        # Energy-based features
        rms_energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features['energy_mean'] = np.mean(rms_energy)
        features['energy_std'] = np.std(rms_energy)
        features['energy_skew'] = skew(rms_energy)
        features['energy_kurtosis'] = kurtosis(rms_energy)
        
        # Rhythm features (tempo and beat tracking)
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = tempo
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                features['rhythm_regularity'] = 1 / (np.std(beat_intervals) + 1e-8)
            else:
                features['rhythm_regularity'] = 0
        except:
            features['tempo'] = 120  # Default tempo
            features['rhythm_regularity'] = 0
        
        return features
    
    def extract_voice_quality_features(self, audio):
        """Extract voice quality features specific to human vs AI speech."""
        features = {}
        
        # Jitter (frequency perturbation)
        f0 = librosa.yin(audio, fmin=50, fmax=400)
        valid_f0 = f0[f0 > 0]
        
        if len(valid_f0) > 1:
            period_diffs = np.abs(np.diff(1/valid_f0))
            features['jitter'] = np.mean(period_diffs) / np.mean(1/valid_f0) * 100
        else:
            features['jitter'] = 0
        
        # Shimmer (amplitude perturbation)
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        if len(rms) > 1:
            amplitude_diffs = np.abs(np.diff(rms))
            features['shimmer'] = np.mean(amplitude_diffs) / np.mean(rms) * 100
        else:
            features['shimmer'] = 0
        
        # Harmonicity (HNR - Harmonic-to-Noise Ratio)
        try:
            # Simple HNR estimation
            stft = librosa.stft(audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Estimate harmonic and noise components
            harmonic = librosa.effects.hpss(audio)[0]
            noise = audio - harmonic
            
            harmonic_energy = np.sum(harmonic**2)
            noise_energy = np.sum(noise**2)
            
            if noise_energy > 0:
                features['hnr'] = 10 * np.log10(harmonic_energy / noise_energy)
            else:
                features['hnr'] = 100  # Very high HNR
        except:
            features['hnr'] = 20  # Default HNR
        
        # Spectral entropy (measure of spectral randomness)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Normalize to get probability distribution
        magnitude_norm = magnitude / (np.sum(magnitude, axis=0) + 1e-8)
        
        # Calculate entropy for each time frame
        entropy_frames = []
        for t in range(magnitude_norm.shape[1]):
            frame = magnitude_norm[:, t]
            frame = frame[frame > 0]  # Remove zeros
            if len(frame) > 0:
                entropy = -np.sum(frame * np.log2(frame + 1e-8))
                entropy_frames.append(entropy)
        
        if entropy_frames:
            features['spectral_entropy_mean'] = np.mean(entropy_frames)
            features['spectral_entropy_std'] = np.std(entropy_frames)
        else:
            features['spectral_entropy_mean'] = 0
            features['spectral_entropy_std'] = 0
        
        return features
    
    def extract_formant_features(self, audio):
        """Extract formant frequencies (vocal tract resonances)."""
        features = {}
        
        try:
            # Pre-emphasis filter
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Window the signal
            windowed = emphasized_audio * np.hanning(len(emphasized_audio))
            
            # LPC analysis to find formants
            lpc_order = int(2 + self.sample_rate / 1000)  # Rule of thumb
            lpc_coeffs = librosa.lpc(windowed, order=lpc_order)
            
            # Find roots of LPC polynomial
            roots = np.roots(lpc_coeffs)
            
            # Extract formant frequencies
            formants = []
            for root in roots:
                if np.imag(root) != 0:  # Complex roots only
                    freq = np.abs(np.angle(root)) * self.sample_rate / (2 * np.pi)
                    if 50 < freq < 4000:  # Typical formant range
                        formants.append(freq)
            
            formants = sorted(formants)[:4]  # Take first 4 formants
            
            # Pad with zeros if not enough formants found
            while len(formants) < 4:
                formants.append(0)
            
            for i, formant in enumerate(formants):
                features[f'formant_f{i+1}'] = formant
                
        except:
            # Default formant values if extraction fails
            for i in range(4):
                features[f'formant_f{i+1}'] = 0
        
        return features
    
    def extract_all_enhanced_features(self, audio):
        """Extract all enhanced features."""
        all_features = {}
        
        # Enhanced MFCC with deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # MFCC statistics
        for i in range(self.n_mfcc):
            all_features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            all_features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            all_features[f'mfcc_{i}_delta_mean'] = np.mean(mfcc_delta[i])
            all_features[f'mfcc_{i}_delta_std'] = np.std(mfcc_delta[i])
            all_features[f'mfcc_{i}_delta2_mean'] = np.mean(mfcc_delta2[i])
            all_features[f'mfcc_{i}_delta2_std'] = np.std(mfcc_delta2[i])
        
        # Enhanced mel-spectrogram
        enhanced_mel = self.extract_enhanced_mel_spectrogram(audio)
        
        # Add other feature sets
        spectral_features = self.extract_advanced_spectral_features(audio)
        prosodic_features = self.extract_prosodic_features(audio)
        voice_quality_features = self.extract_voice_quality_features(audio)
        formant_features = self.extract_formant_features(audio)
        
        # Combine all features
        all_features.update(spectral_features)
        all_features.update(prosodic_features)
        all_features.update(voice_quality_features)
        all_features.update(formant_features)
        
        return all_features, enhanced_mel, mfcc
    
    def create_augmented_dataset(self, features, spectrograms, labels, augmentation_factor=2):
        """Create augmented dataset by applying various transformations."""
        print(f"Creating augmented dataset with factor {augmentation_factor}...")
        
        augmented_features = []
        augmented_spectrograms = []
        augmented_labels = []
        
        # Original data
        augmented_features.extend(features)
        augmented_spectrograms.extend(spectrograms)
        augmented_labels.extend(labels)
        
        # Generate augmented versions
        for i in range(len(features)):
            for aug_idx in range(augmentation_factor):
                # Spectrogram augmentation
                aug_spec = self._augment_spectrogram(spectrograms[i])
                
                # Feature augmentation (add small noise)
                aug_feat = features[i] + np.random.normal(0, 0.01, len(features[i]))
                
                augmented_features.append(aug_feat)
                augmented_spectrograms.append(aug_spec)
                augmented_labels.append(labels[i])
        
        print(f"✓ Augmented dataset created: {len(augmented_features)} samples")
        return np.array(augmented_features), np.array(augmented_spectrograms), np.array(augmented_labels)
    
    def _augment_spectrogram(self, spectrogram):
        """Apply augmentation to spectrogram."""
        aug_spec = spectrogram.copy()
        
        # Time masking
        if np.random.random() < 0.3:
            time_mask_size = np.random.randint(1, min(10, aug_spec.shape[1] // 8))
            time_start = np.random.randint(0, aug_spec.shape[1] - time_mask_size)
            aug_spec[:, time_start:time_start+time_mask_size] = aug_spec.mean()
        
        # Frequency masking
        if np.random.random() < 0.3:
            freq_mask_size = np.random.randint(1, min(15, aug_spec.shape[0] // 8))
            freq_start = np.random.randint(0, aug_spec.shape[0] - freq_mask_size)
            aug_spec[freq_start:freq_start+freq_mask_size, :] = aug_spec.mean()
        
        # Gaussian noise
        if np.random.random() < 0.2:
            noise_factor = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_factor, aug_spec.shape)
            aug_spec += noise
        
        return aug_spec


class AdvancedDatasetProcessor:
    """Advanced dataset processor with enhanced preprocessing."""
    
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.label_mapping = {'real': 0, 'fake': 1}
    
    def process_dataset_enhanced(self, split, augment=False, balance_classes=True):
        """Process dataset with enhanced features and optional augmentation."""
        print(f"\\nProcessing {split} dataset (enhanced)...")
        
        features_list = []
        spectrograms_list = []
        labels_list = []
        paths_list = []
        
        for class_name, label in self.label_mapping.items():
            class_dir = os.path.join(self.data_dir, split, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            print(f"  Processing {len(audio_files)} {class_name} files...")
            
            for audio_file in audio_files:
                file_path = os.path.join(class_dir, audio_file)
                
                # Load audio with augmentation for training
                should_augment = augment and split == 'training'
                audio = self.feature_extractor.load_audio_enhanced(
                    file_path, duration=3.0, augment=should_augment
                )
                
                if audio is not None:
                    try:
                        # Extract enhanced features
                        features, enhanced_mel, mfcc = self.feature_extractor.extract_all_enhanced_features(audio)
                        
                        # Convert features dict to array
                        feature_array = np.array(list(features.values()))
                        
                        features_list.append(feature_array)
                        spectrograms_list.append(enhanced_mel)
                        labels_list.append(label)
                        paths_list.append(file_path)
                        
                    except Exception as e:
                        print(f"    Error processing {audio_file}: {e}")
                        continue
        
        features_array = np.array(features_list)
        spectrograms_array = np.array(spectrograms_list)
        labels_array = np.array(labels_list)
        
        print(f"  Processed {len(features_array)} files successfully")
        print(f"  Feature shape: {features_array.shape}")
        print(f"  Spectrogram shape: {spectrograms_array.shape}")
        
        # Balance classes if requested
        if balance_classes and split == 'training':
            features_array, spectrograms_array, labels_array = self._balance_classes(
                features_array, spectrograms_array, labels_array
            )
        
        return features_array, spectrograms_array, labels_array, paths_list
    
    def _balance_classes(self, features, spectrograms, labels):
        """Balance classes by oversampling minority class."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(unique_labels) != 2:
            return features, spectrograms, labels
        
        max_count = np.max(counts)
        min_count = np.min(counts)
        
        print(f"  Balancing classes: {counts[0]} real, {counts[1]} fake")
        
        # Find minority class
        minority_class = unique_labels[np.argmin(counts)]
        minority_indices = np.where(labels == minority_class)[0]
        
        # Calculate how many samples to add
        samples_to_add = max_count - min_count
        
        # Randomly sample from minority class with replacement
        additional_indices = np.random.choice(minority_indices, samples_to_add, replace=True)
        
        # Combine original and additional samples
        balanced_features = np.vstack([features, features[additional_indices]])
        balanced_spectrograms = np.vstack([spectrograms, spectrograms[additional_indices]])
        balanced_labels = np.hstack([labels, labels[additional_indices]])
        
        print(f"  Balanced dataset: {len(balanced_features)} samples")
        
        return balanced_features, balanced_spectrograms, balanced_labels


if __name__ == "__main__":
    print("Testing Enhanced Audio Preprocessing...")
    
    # Test enhanced feature extractor
    extractor = EnhancedAudioFeatureExtractor()
    
    data_dir = "/Users/berkut/Desktop/Projects/deepfakedeneme/model"
    test_file = None
    
    # Find a test file
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    if test_file:
        print(f"\\nTesting with file: {os.path.basename(test_file)}")
        
        # Load and process audio
        audio = extractor.load_audio_enhanced(test_file, augment=True)
        
        if audio is not None:
            print(f"✓ Audio loaded: {audio.shape}")
            
            # Extract enhanced features
            features, enhanced_mel, mfcc = extractor.extract_all_enhanced_features(audio)
            
            print(f"✓ Features extracted: {len(features)} features")
            print(f"✓ Enhanced mel-spectrogram: {enhanced_mel.shape}")
            print(f"✓ MFCC: {mfcc.shape}")
            
            print("\\n📊 Sample features:")
            for i, (key, value) in enumerate(list(features.items())[:10]):
                print(f"  {key}: {value:.4f}")
            
        print("\\n✅ Enhanced preprocessing test completed!")
    else:
        print("\\n❌ No test audio file found!")
