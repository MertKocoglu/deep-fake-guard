#!/usr/bin/env python3
"""
Audio Preprocessing Module for Deepfake Detection
=================================================

This module handles the preprocessing of audio files including:
- Feature extraction (MFCCs, spectrograms, pitch, jitter, shimmer)
- Mel-spectrogram conversion
- Data normalization and augmentation

Author: ML Engineer
Date: June 2025
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract comprehensive audio features for deepfake detection."""
    
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=13, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        
    def load_audio(self, file_path, duration=2.0):
        """Load audio file with specified duration."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            # Ensure audio is exactly 2 seconds
            target_length = int(self.sample_rate * duration)
            if len(audio) < target_length:
                # Pad with zeros if shorter
                audio = np.pad(audio, (0, target_length - len(audio)))
            elif len(audio) > target_length:
                # Truncate if longer
                audio = audio[:target_length]
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel-spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, audio):
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_spectral_features(self, audio):
        """Extract spectral features like spectral centroid, rolloff, etc."""
        features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        return features
    
    def extract_pitch_features(self, audio):
        """Extract pitch-related features."""
        try:
            # Extract fundamental frequency (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Remove NaN values
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                pitch_mean = np.mean(f0_clean)
                pitch_std = np.std(f0_clean)
                pitch_min = np.min(f0_clean)
                pitch_max = np.max(f0_clean)
                
                # Jitter (pitch period variability)
                if len(f0_clean) > 1:
                    pitch_periods = 1.0 / f0_clean
                    jitter = np.std(pitch_periods) / np.mean(pitch_periods) * 100
                else:
                    jitter = 0
            else:
                pitch_mean = pitch_std = pitch_min = pitch_max = jitter = 0
                
            return {
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'pitch_min': pitch_min,
                'pitch_max': pitch_max,
                'jitter': jitter,
                'voiced_fraction': np.mean(voiced_flag)
            }
        except Exception as e:
            print(f"Error extracting pitch features: {e}")
            return {
                'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0,
                'pitch_max': 0, 'jitter': 0, 'voiced_fraction': 0
            }
    
    def extract_shimmer(self, audio):
        """Extract shimmer (amplitude variability) features."""
        try:
            # Extract short-time energy
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            # Calculate RMS energy for each frame
            rms = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            if len(rms) > 1:
                # Shimmer is the relative variability in amplitude
                shimmer = np.std(rms) / np.mean(rms) * 100 if np.mean(rms) > 0 else 0
            else:
                shimmer = 0
                
            return {
                'shimmer': shimmer,
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms)
            }
        except Exception as e:
            print(f"Error extracting shimmer: {e}")
            return {'shimmer': 0, 'rms_mean': 0, 'rms_std': 0}
    
    def extract_all_features(self, audio):
        """Extract all features from audio signal."""
        features = {}
        
        # Mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # MFCC
        mfcc = self.extract_mfcc(audio)
        
        # Statistical features from MFCC
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # Pitch features
        pitch_features = self.extract_pitch_features(audio)
        features.update(pitch_features)
        
        # Shimmer features
        shimmer_features = self.extract_shimmer(audio)
        features.update(shimmer_features)
        
        return features, mel_spec, mfcc

class DatasetProcessor:
    """Process the entire dataset for training."""
    
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        
    def process_dataset(self, split='training'):
        """Process a dataset split (training, validation, testing)."""
        split_dir = os.path.join(self.data_dir, split)
        
        # Get file paths
        real_files = []
        fake_files = []
        
        real_dir = os.path.join(split_dir, 'real')
        fake_dir = os.path.join(split_dir, 'fake')
        
        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]
        
        print(f"Processing {split} set:")
        print(f"  Real files: {len(real_files)}")
        print(f"  Fake files: {len(fake_files)}")
        
        # Process files
        features_list = []
        mel_spectrograms = []
        labels = []
        file_paths = []
        
        # Process real files
        for file_path in tqdm(real_files, desc="Processing real files"):
            audio = self.feature_extractor.load_audio(file_path)
            if audio is not None:
                features, mel_spec, mfcc = self.feature_extractor.extract_all_features(audio)
                features_list.append(features)
                mel_spectrograms.append(mel_spec)
                labels.append(0)  # Real = 0
                file_paths.append(file_path)
        
        # Process fake files
        for file_path in tqdm(fake_files, desc="Processing fake files"):
            audio = self.feature_extractor.load_audio(file_path)
            if audio is not None:
                features, mel_spec, mfcc = self.feature_extractor.extract_all_features(audio)
                features_list.append(features)
                mel_spectrograms.append(mel_spec)
                labels.append(1)  # Fake = 1
                file_paths.append(file_path)
        
        # Convert to DataFrames and arrays
        features_df = pd.DataFrame(features_list)
        mel_spectrograms = np.array(mel_spectrograms)
        labels = np.array(labels)
        
        print(f"Processed {len(features_df)} files successfully")
        print(f"Feature shape: {features_df.shape}")
        print(f"Mel-spectrogram shape: {mel_spectrograms.shape}")
        
        return features_df, mel_spectrograms, labels, file_paths

def save_mel_spectrogram_image(mel_spec, output_path, title="Mel-spectrogram"):
    """Save mel-spectrogram as image."""
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spec, sr=16000, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    data_dir = "/Users/berkut/Desktop/Projects/deepfakedeneme/model"
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    
    # Initialize dataset processor
    processor = DatasetProcessor(data_dir, feature_extractor)
    
    # Process training set as example
    features_df, mel_specs, labels, file_paths = processor.process_dataset('training')
    
    print("\nDataset processing completed!")
    print(f"Features shape: {features_df.shape}")
    print(f"Mel-spectrograms shape: {mel_specs.shape}")
    print(f"Labels distribution: Real={np.sum(labels==0)}, Fake={np.sum(labels==1)}")
