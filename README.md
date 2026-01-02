# 🎵 Deepfake Audio Detection System

AI-powered deepfake audio detection system with advanced CNN model achieving 90% accuracy and modern web interface.

## 🏆 Features

- **Advanced CNN Model**: 90% accuracy, 89.8% F1-score
- **Modern Web Interface**: Drag & drop file upload, real-time analysis
- **Multiple Audio Formats**: WAV, MP3, FLAC, M4A, OGG support
- **Visual Analysis**: Mel-spectrogram and waveform visualization
- **Report Generation**: Detailed analysis reports
- **Production Ready**: Flask web application with REST API

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip >= 21.0
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd deepfakedeneme/model

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app_simple.py
```

### Access
```
🌐 Web Interface: http://localhost:5000
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
1. Open http://localhost:5000
2. Drag & drop audio file or click to browse
3. Click "Analyze Audio"
4. View results and download report

### Python API
```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('results/best_advanced_cnn_model.h5')

# Predict (with optimal threshold)
predictions = model.predict(spectrograms)
is_fake = predictions > 0.0001  # Optimal threshold
```

## 📁 Project Structure

```
model/
├── app_simple.py          # Main web application
├── cnn_model.py           # CNN model architectures
├── audio_preprocessing.py # Audio processing utilities
├── main_pipeline.py       # Training pipeline
├── results/               # Trained models and outputs
├── templates/             # HTML templates
├── static/                # CSS, JS, images
└── requirements.txt       # Dependencies
```

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

### Web Technology
- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5 + Custom CSS/JS
- **API**: RESTful endpoints
- **File Handling**: Secure upload with validation

## 🔧 Development

### Training New Models
```bash
python main_pipeline.py
```

### Testing
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

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_simple:app
```

## 📝 API Documentation

### Endpoints

#### `POST /upload`
Upload and analyze audio file.

**Request**: Multipart form with audio file
**Response**: JSON with detection results

#### `POST /generate_report`
Generate analysis report.

**Request**: JSON with analysis data
**Response**: Report download URL

## ⚠️ Important Notes

- **Optimal Threshold**: Always use 0.0001 (not 0.5) for 90% accuracy
- **File Size**: Maximum 16MB per upload
- **Formats**: WAV, MP3, FLAC, M4A, OGG supported
- **Duration**: Optimal performance with 2-second audio clips

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

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
