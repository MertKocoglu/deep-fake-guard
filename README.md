# 🎵 DeepFake Guard - AI Audio Detection System

Advanced deepfake audio detection system powered by CNN achieving **90% accuracy**. Modern React SPA with real-time analysis, microphone recording, and comprehensive reporting.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)
![React](https://img.shields.io/badge/react-18.2-blue.svg)

## ✨ Features

### 🎯 Core Capabilities
- **🤖 Advanced CNN Model** - 90% accuracy, 89.8% F1-score
- **⚛️ Modern React Interface** - Single Page Application with Vite
- **🎤 Microphone Recording** - Record and analyze audio in-browser
- **📁 Multi-Format Support** - WAV, MP3, FLAC, M4A, OGG, WEBM
- **🔄 WebM Auto-Conversion** - FFmpeg integration for Chrome compatibility
- **📊 Visual Analysis** - Mel-spectrogram and waveform visualization
- **📄 Report Generation** - Detailed TXT reports with technical metrics
- **🔊 Audio Playback** - Built-in player for recorded/uploaded files
- **👤 User Authentication** - Secure login/register system
- **📜 Analysis History** - Track all analyses per user
- **🐘 PostgreSQL Database** - Production-ready database system

### ⚙️ Technical Features
- **REST API** - Flask backend with JSON responses
- **Optimized Threshold** - Fine-tuned detection (0.0001)
- **Responsive Design** - Bootstrap 5 + React components
- **Modern Tooling** - Vite for fast builds and HMR
- **Auto Cleanup** - Automatic file management
- **Secure Sessions** - HTTPOnly cookies, CSRF protection

## 🚀 Quick Start

### 📋 Prerequisites
```bash
Python >= 3.11
PostgreSQL >= 14
Node.js >= 18 (for frontend development)
FFmpeg (for audio conversion)
```

### 📦 Installation

#### 1️⃣ Clone Repository
```bash
git clone <repository-url>
cd "deepfake guard"
```

#### 2️⃣ Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 3️⃣ Configure Database
```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE deepfake_guard;"

# Configure .env file
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Initialize database tables
python migrate_to_postgres.py
```

#### 4️⃣ Install FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows - Download from https://ffmpeg.org/download.html
```

#### 5️⃣ Run Application
```bash
python app.py
```

### 🌐 Access Points
- **Main App**: http://localhost:8080/
- **Login**: http://localhost:8080/login
- **Register**: http://localhost:8080/register

### 👥 Default Users
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Administrator |
| demo | demo123 | Demo User |
| user | user123 | Standard User |

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **90.0%** |
| **F1-Score** | **89.8%** |
| **Precision** | 87.9% |
| **Recall** | 92.1% |
| **ROC-AUC** | 0.966 |

### 📈 Dataset
- **Training**: 13,957 samples
- **Validation**: 2,826 samples
- **Testing**: 1,089 samples
- **Total**: 17,872 files (balanced 50/50 real/fake)

## 🎯 Usage Guide

### 🌐 Web Interface

1. **Login** - Use default credentials or register new account
2. **Upload Audio** (Option A):
   - Drag & drop file or click to browse
   - Supports: WAV, MP3, FLAC, M4A, OGG, WEBM
   - Max size: 16MB
3. **Record Audio** (Option B):
   - Click "Start Recording"
   - Speak into microphone
   - Click "Stop Recording"
4. **Analyze** - Click "Analyze Speech"
5. **Review Results**:
   - View prediction (REAL/FAKE)
   - Check confidence score
   - See spectrogram visualization
   - Play audio
   - Download detailed report

### 🐍 Python API

```python
import tensorflow as tf
import librosa
import numpy as np

# Load model
model = tf.keras.models.load_model('results/best_advanced_cnn_model.h5')

# Process audio
audio, sr = librosa.load('audio.wav', sr=16000, duration=2.0)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_spec = np.expand_dims(mel_spec, axis=-1)
mel_spec = np.expand_dims(mel_spec, axis=0)

# Predict
prediction = model.predict(mel_spec)[0][0]
threshold = 0.0001
is_fake = prediction > threshold
confidence = abs(prediction - threshold) * 10000

print(f"Result: {'DEEPFAKE' if is_fake else 'REAL'}")
print(f"Confidence: {confidence:.1f}%")
```

## 📁 Project Structure

```
deepfake guard/
├── 🐍 Backend
│   ├── app.py                      # Main Flask application
│   ├── database.py                 # SQLAlchemy models
│   ├── clear_users.py              # Database utility
│   ├── migrate_to_postgres.py      # DB migration script
│   └── .env                        # Environment variables
│
├── 🧠 ML Model
│   └── results/
│       ├── best_advanced_cnn_model.h5  # Trained model (90%)
│       └── advanced_cnn_90percent_accuracy.h5
│
├── ⚛️ Frontend (React + Vite)
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx             # Main component
│       │   ├── App.css             # Styles
│       │   └── main.jsx            # Entry point
│       ├── package.json
│       └── vite.config.js
│
├── 🎨 Templates
│   ├── templates/
│   │   ├── react_app.html          # React wrapper
│   │   ├── login.html              # Login page
│   │   └── register.html           # Register page
│   └── static/
│       ├── logo.png                # Application logo
│       └── react-dist/             # Built React assets
│
├── 📁 Data Directories
│   ├── training/                   # Training audio files
│   ├── validation/                 # Validation audio files
│   ├── testing/                    # Testing audio files
│   └── uploads/                    # Temporary uploads
│
└── 📄 Documentation
    ├── README.md                   # This file
    ├── requirements.txt            # Python dependencies
    └── .env.example                # Environment template
```

## 🔧 Configuration

### 🗄️ Database Setup

Create `.env` file:
```bash
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/deepfake_guard
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
DEBUG=True
```

Initialize database:
```bash
psql -U postgres -c "CREATE DATABASE deepfake_guard;"
python migrate_to_postgres.py
```

### 🎥 FFmpeg Configuration

Required for WebM format support (Chrome recordings):
```bash
# Verify installation
ffmpeg -version

# Should output FFmpeg version info
```

## 🛠️ Development

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Development server (HMR enabled)
npm run dev

# Production build
npm run build
```

### Database Management
```bash
# Clear all users
python clear_users.py

# Recreate tables
python migrate_to_postgres.py
```

### Debug Mode
```bash
# Flask debug mode enabled by default
python app.py
```

## 🧠 Technical Architecture

### Model Details
- **Architecture**: Advanced CNN with residual connections
- **Input Shape**: 128x63x1 (mel-spectrogram)
- **Layers**: Conv2D, BatchNorm, MaxPooling, Dense
- **Activation**: ReLU, Sigmoid output
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

### Audio Processing Pipeline
1. **Load**: Read audio file (librosa)
2. **Resample**: Convert to 16kHz
3. **Trim**: Extract first 2 seconds
4. **Transform**: Generate mel-spectrogram (128 mel bins)
5. **Normalize**: Min-max scaling
6. **Predict**: CNN inference
7. **Threshold**: Apply optimal threshold (0.0001)

### API Endpoints

#### `POST /upload`
Upload and analyze audio file
- **Input**: Multipart form data with audio file
- **Output**: JSON with prediction results

#### `POST /generate_report`
Generate detailed analysis report
- **Input**: JSON with analysis metadata
- **Output**: TXT file with technical details

#### `GET /downloads/<filename>`
Serve uploaded audio for playback
- **Input**: Filename parameter
- **Output**: Audio file stream

## 🚀 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Environment Variables
```bash
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
export SECRET_KEY=production-secret-key
export FLASK_ENV=production
export DEBUG=False
```

### Nginx Configuration (Optional)
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## ⚠️ Important Notes

- **Audio Length**: System processes first 2 seconds of audio
- **File Cleanup**: Uploaded files are automatically deleted after analysis
- **Session Timeout**: 24 hours by default
- **Max Upload Size**: 16MB per file
- **Supported Browsers**: Chrome, Firefox, Safari, Edge (latest versions)

## 🔒 Security Features

- ✅ File validation and sanitization
- ✅ Secure filename handling
- ✅ SQL injection protection (SQLAlchemy ORM)
- ✅ XSS protection (Flask auto-escaping)
- ✅ CSRF protection (session tokens)
- ✅ HTTPOnly cookies
- ✅ Password hashing (Werkzeug)

## 🐛 Troubleshooting

### Database Connection Error
```bash
# Check PostgreSQL is running
brew services list  # macOS
sudo systemctl status postgresql  # Linux

# Verify credentials in .env
psql -U postgres -d deepfake_guard
```

### FFmpeg Not Found
```bash
# Add to PATH or install
which ffmpeg  # Should return path
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🙏 Acknowledgments

- **TensorFlow** - Deep learning framework
- **Librosa** - Audio processing library
- **Flask** - Web framework
- **React** - UI library
- **Vite** - Build tool
- **Bootstrap** - CSS framework
- **PostgreSQL** - Database system

---

**Built with ❤️ for combating deepfake audio**  
**Last Updated:** January 16, 2026

## 🏆 Features

### Core Features
- **Advanced CNN Model**: 90% accuracy, 89.8% F1-score
- **Modern React Interface**: Single Page Application with responsive design
- **Microphone Recording**: Record and analyze audio directly in browser
- **Multiple Audio Formats**: WAV, MP3, FLAC, M4A, OGG, WEBM support
- **WebM Support**: Automatic conversion using FFmpeg
- **Real-time Analysis**: Live processing with progress indicators
- **Visual Analysis**: Mel-spectrogram and waveform visualization
- **Report Generation**: TXT format reports with technical details
- **Audio Playback**: Built-in player for uploaded/recorded audio
- **User Authentication**: Secure login/register system
- **Analysis History**: Track all analyses per user
- **PostgreSQL Database**: Production-ready database system

### Technical Features
- **Production Ready**: Flask REST API backend
- **Optimal Threshold**: Fine-tuned detection threshold (0.0001)
- **Responsive Design**: Bootstrap 5 + React
- **React + Vite**: Modern frontend build tooling
- **Auto File Cleanup**: Automatic uploaded file management
- **Session Management**: Secure user sessions

## 🚀 Quick Start

### Prerequisites
```bash
Python >= 3.11
Node.js >= 18 (for React development)
FFmpeg (for audio conversion)
pip >= 23.0
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd "deepfake guard"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install Python dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```

### React Development (Optional)
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Development mode
npm run dev

# Build for production
npm run build
```

### Access
```
🌐 Web Interface: http://localhost:8080/
🔐 Login Page: http://localhost:8080/login
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
2. Login with default credentials or register
3. **Option A - Upload File:**
   - Drag & drop audio file or click to browse
   - Supported formats: WAV, MP3, FLAC, M4A, OGG, WEBM
4. **Option B - Record Audio:**
   - Click "Start Recording"
   - Speak into microphone
   - Click "Stop Recording"
5. Click "Analyze Speech"
6. View results with spectrogram visualization
7. Play audio, download report, or analyze another

### Default Users
```
Username: demo / Password: demo123
Username: admin / Password: admin123
Username: mert / Password: mert123
```

### Python API
```python
import tensorflow as tf
import numpy as np
import librosa

# Load model and scaler
model = tf.keras.models.load_model('results/best_advanced_cnn_model.h5')

# Load and process audio
audio, sr = librosa.load('audio.wav', sr=16000)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

# Predict with optimal threshold
prediction = model.predict(mel_spec)
is_fake = prediction > 0.0001  # Optimal threshold
confidence = abs(prediction - 0.0001) * 100
```

## 📁 Project Structure

```
deepfake guard/
├── app.py                          # Main Flask application
├── database.py                     # Database models & setup
├── clear_users.py                  # Utility to clear database users
├── results/
│   ├── best_advanced_cnn_model.h5 # Trained model (90% accuracy)
│   ├── feature_scaler.pkl         # Feature scaler
│   └── *.pkl                      # Other model artifacts
├── frontend/                       # React application
│   ├── src/
│   │   ├── App.jsx                # Main React component
│   │   ├── App.css                # React styles
│   │   └── main.jsx               # React entry point
│   ├── package.json               # Node dependencies
│   └── vite.config.js             # Vite configuration
├── templates/                      # Flask templates
│   ├── index.html                 # Traditional interface
│   ├── react_app.html             # React wrapper
│   ├── login.html                 # Login page
│   └── register.html              # Registration page
├── static/
│   ├── react_app.html             # Main React wrapper
│   ├── login.html                 # Login page
│   └── register.html              # Registration page
├── static/
│   ├── css/style.css              # Shared styles
└── README.md                      # This file
```

## 🔧 Configuration

### Database
The application uses PostgreSQL. Configure your database connection in `.env` file:

```bash
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/deepfake_guard
```

Create the database:
```bash
psql -U postgres -c "CREATE DATABASE deepfake_guard;"
python migrate_to_postgres.py
```

### FFmpeg
FFmpeg is required for WebM audio conversion:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 🛠️ Development

### Clear Database
```bash
python clear_users.py
```

### Rebuild React App
```bash
cd frontend
npm run build
```

### Run in Development Mode
```bash
# Flask debug mode is enabled by default
python app.py
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
