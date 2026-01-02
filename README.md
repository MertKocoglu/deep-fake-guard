# 🎵 Deepfake Audio Detection System

AI-powered deepfake audio detection system with advanced CNN model achieving **90% accuracy** and modern web interface with PDF report generation.

## 🏆 Features

- **Advanced CNN Model**: 90% accuracy, 89.8% F1-score
- **Modern Web Interface**: Drag & drop file upload, real-time analysis
- **PDF Reports**: Professional analysis reports with visualizations
- **Multiple Audio Formats**: WAV, MP3, FLAC, M4A, OGG support
- **Visual Analysis**: Mel-spectrogram and waveform visualization
- **Production Ready**: Flask web application with REST API
- **Optimal Threshold**: Fine-tuned detection threshold (0.0001)

## 🚀 Quick Start

### Prerequisites
```bash
Python >= 3.10
pip >= 23.0
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd "deepfake guard"

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```

### Access
```
🌐 Web Interface: http://localhost:8080
📱 Network Access: http://YOUR_IP:8080
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **90.0%** |
| **F1-Score** | **89.8%** |
| **Precision** | 87.9% |
| **Recall** | 92.1% |
| **ROC-AUC** | 0.966 |

## 🎯 Usage

### Web Interface
1. Open http://localhost:8080 in your browser
2. Drag & drop audio file or click to browse
3. Click "Analyze Audio"
4. View real-time results with visualizations
5. Generate and download PDF report

### Python API
```python
import tensorflow as tf
import numpy as np
import librosa

# Load model and scaler
model = tf.keras.models.load_model('results/best_advanced_cnn_model.h5')

# Load and process audio
audio, sr = librosa.load('audio.wav', sr=16000, duration=2.0)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

# Predict with optimal threshold
prediction = model.predict(mel_spec)
is_fake = prediction > 0.0001  # Optimal threshold
confidence = abs(prediction - 0.0001) * 100
```

## 📁 Project Structure

```
deepfake guard/
├── app.py                          # Main web application (with PDF)
├── app_simple.py                   # Web app without PDF
├── results/
│   ├── best_advanced_cnn_model.h5 # Trained model (90% accuracy)
│   └── feature_scaler.pkl         # Feature scaler
├── templates/
│   └── index.html                 # Web interface
├── static/
│   ├── css/style.css              # Styling
│   └── js/main.js                 # Client-side logic
├── uploads/                       # Temporary upload folder
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

**Note:** Training data, model training scripts, and development files are not included in production deployment.

## 🧠 Technical Details

### Model Architecture
- **Type**: Advanced Convolutional Neural Network
- **Input**: Mel-spectrograms (128x63x1)
- **Features**: Residual connections, attention mechanisms
- **Optimization**: Adam optimizer with optimal threshold (0.0001)

### Audio Processing
- **Sample Rate**: 16 kHz
- **Duration**: 2 seconds
- **Features**: Mel-spectrograms with 128 mel bins
- **Normalization**: Min-max scaling

### Running in Development Mode
```bash
# With debug enabled
python app.py
```

### Dataset Information
- **Training samples**: 13,957 audio files
- **Validation samples**: 2,826 audio files
- **Test samples**: 1,089 audio files
- **Total dataset**: 17,872 audio files (50% real, 50% fake) Testing
```bash
python test_pipeline.py
```

### Model Analysis
```bash
python complete_analysis.py
```

## 📈 Results

The system has been tested on a balanced dataset and achieves:
- **90% accuracy** on test data
- **Real audio detection**: 92.1%
- **Fake audio detection**: 87.9%
- **Production ready** performance

## 🚀 Deployment

### Local Development
```bash
python app_simple.py
```

### Produc.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Docker Deployment (Optional)
```bash
# Build image
docker build -t deepfake-detector .

# Run container
docker run -p 8080:8080 deepfake-detector

### Endpoints

#### `POST /upload`
Upload and analyze audio file.

**Request**: Multipart form with audio file
**Response**: JSON with detection results

#### `POST /generate_report`
Generate analysis report.

**Request**: JSON with analysis data
**ReSupported Formats**: WAV, MP3, FLAC, M4A, OGG
- **Audio Duration**: Processes first 2 seconds of audio
- **Python Version**: Requires Python 3.10+ (tested on 3.13)
- **Dependencies**: All required packages in `requirements.txt`

## 🛠️ System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 1GB free space
- **CPU**: Multi-core processor recommended

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 2GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster inference)
## ⚠️ Important Notes

- **TensorFlow** - Deep learning framework
- **Librosa** - Audio processing library
- **Flask** - Web framework
- **ReportLab** - PDF generation
- **Bootstrap** - UI framework

## 🔒 Security

- File validation and sanitization
- Secure file upload handling
- No data persistence (files deleted after processing)
- XSS and CSRF protection

## 📅 Version History

- **v1.0.0** (January 2026) - Initial release with 90% accuracy
  - Advanced CNN model
  - Web interface with PDF reports
  - Real-time audio analysis

---

**Built with ❤️ for combating deepfake audio**

**Last Updated:** January 2, 2026 - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For questions and support, please open an issue in the repository.

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- Librosa team for audio processing
- Flask team for the web framework
- Bootstrap team for the UI framework

---

**Built with ❤️ for combating deepfake audio**
