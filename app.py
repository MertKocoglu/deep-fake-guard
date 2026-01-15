#!/usr/bin/env python3
"""
Deepfake Audio Detection Web Interface
=====================================

Flask web application for deepfake audio detection with file upload,
real-time detection, and report generation.

Author: ML Engineer
Date: 24 Haziran 2025
"""

import os
import uuid
import subprocess
import librosa
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, redirect, url_for, session
from werkzeug.utils import secure_filename
from functools import wraps
import pickle
from database import db, User, AnalysisHistory, init_db
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Load environment variables
load_dotenv()

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("ReportLab not available. PDF reports will be disabled.")
    REPORTLAB_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Database configuration
# PostgreSQL connection
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'postgresql://postgres:postgres@localhost:5432/deepfake_guard'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
init_db(app)

# Global variables for model and scaler
model = None
scaler = None

def login_required(f):
    """Decorator to require login for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def load_model_and_scaler():
    """Load the trained model and scaler."""
    global model, scaler
    
    try:
        print("Loading trained model...")
        model_path = "results/best_advanced_cnn_model.h5"
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        print("Loading feature scaler...")
        scaler_path = "results/feature_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        
        return True
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return False

def allowed_file(filename):
    """Check if uploaded file is allowed."""
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path):
    """Convert audio file to WAV format using ffmpeg if needed."""
    # Check if file is already WAV
    if input_path.lower().endswith('.wav'):
        return input_path
    
    # Create output path
    output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    
    try:
        import subprocess
        # Convert using ffmpeg
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"Converted {input_path} to WAV format")
            return output_path
        else:
            print(f"FFmpeg conversion failed: {result.stderr}")
            return input_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return input_path

def extract_audio_features(file_path):
    """Extract mel-spectrogram from audio file."""
    try:
        # Convert to WAV if needed (for WebM and other formats)
        wav_path = convert_to_wav(file_path)
        
        # Load audio file (no duration limit) - keep original for duration calculation
        audio_original, sr = librosa.load(wav_path, sr=16000)
        original_duration = len(audio_original) / sr  # Calculate original duration
        
        # Copy audio for processing
        audio = audio_original.copy()
        
        # Ensure audio is exactly 4 seconds for processing
        target_length = 4 * 16000
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Truncate if too long
            audio = audio[:target_length]
        
        # Extract mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )
        
        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Ensure shape is (128, 63)
        if mel_spectrogram_db.shape[1] != 63:
            if mel_spectrogram_db.shape[1] > 63:
                mel_spectrogram_db = mel_spectrogram_db[:, :63]
            else:
                # Pad with minimum values
                pad_width = 63 - mel_spectrogram_db.shape[1]
                mel_spectrogram_db = np.pad(
                    mel_spectrogram_db, 
                    ((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=mel_spectrogram_db.min()
                )
        
        # Normalize
        mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.min()) / \
                           (mel_spectrogram_db.max() - mel_spectrogram_db.min())
        
        # Add channel dimension for CNN
        mel_spectrogram_db = mel_spectrogram_db[..., np.newaxis]
        
        return mel_spectrogram_db, audio_original, sr, original_duration
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def predict_deepfake(spectrogram):
    """Predict if audio is deepfake using the trained model."""
    try:
        # Add batch dimension
        spectrogram_batch = np.expand_dims(spectrogram, axis=0)
        
        # Predict
        prediction_proba = model.predict(spectrogram_batch, verbose=0)[0][0]
        
        # Use optimal threshold
        optimal_threshold = 0.0001
        is_fake = prediction_proba > optimal_threshold
        
        # Calculate confidence
        if is_fake:
            confidence = min(prediction_proba * 10000, 99.9)  # Scale up from very small values
        else:
            confidence = min((1 - prediction_proba) * 100, 99.9)
        
        return {
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'raw_probability': float(prediction_proba),
            'threshold_used': optimal_threshold
        }
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def create_spectrogram_plot(spectrogram, audio, sr):
    """Create spectrogram visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot waveform
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax1.plot(time, audio)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot spectrogram
    spectrogram_2d = spectrogram.squeeze()  # Remove channel dimension
    im = ax2.imshow(spectrogram_2d, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('Mel-Spectrogram')
    ax2.set_xlabel('Time Frames')
    ax2.set_ylabel('Mel Frequency Bins')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    # Convert to base64 for web display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def generate_pdf_report(result_data):
    """Generate PDF report of the detection results."""
    filename = f"deepfake_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.darkblue
    )
    story.append(Paragraph("DEEPFAKE AUDIO DETECTION REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Report info
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=10
    )
    
    report_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", info_style))
    story.append(Paragraph(f"<b>File Name:</b> {result_data['filename']}", info_style))
    story.append(Paragraph(f"<b>Analysis Model:</b> Advanced CNN (90% Accuracy)", info_style))
    story.append(Spacer(1, 20))
    
    # Results section
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        textColor=colors.darkgreen if not result_data['is_fake'] else colors.darkred
    )
    
    result_text = "REAL AUDIO DETECTED" if not result_data['is_fake'] else "FAKE AUDIO DETECTED"
    story.append(Paragraph(result_text, result_style))
    
    # Detailed results table
    data = [
        ['Metric', 'Value'],
        ['Classification', 'Real Audio' if not result_data['is_fake'] else 'Deepfake Audio'],
        ['Confidence Level', f"{result_data['confidence']:.1f}%"],
        ['Raw Probability', f"{result_data['raw_probability']:.6f}"],
        ['Threshold Used', f"{result_data['threshold_used']:.6f}"],
        ['Model Accuracy', '90.0%'],
        ['Model Type', 'Advanced CNN'],
    ]
    
    table = Table(data, colWidths=[2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 30))
    
    # Technical details
    tech_title = ParagraphStyle(
        'TechTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    )
    story.append(Paragraph("Technical Details", tech_title))
    
    tech_text = f"""
    <b>Analysis Method:</b> The audio file was processed using advanced machine learning techniques.
    The system extracts mel-spectrogram features and analyzes them using a trained Convolutional Neural Network (CNN).
    <br/><br/>
    <b>Model Performance:</b> Our Advanced CNN model achieves 90% accuracy with an F1-score of 89.8% on test data.
    The model uses an optimized threshold of 0.0001 for classification decisions.
    <br/><br/>
    <b>Confidence Interpretation:</b> The confidence level indicates how certain the model is about its prediction.
    Higher confidence values suggest more reliable results.
    <br/><br/>
    <b>Note:</b> This analysis is based on the current state-of-the-art deepfake detection technology.
    For critical applications, we recommend additional verification methods.
    """
    
    story.append(Paragraph(tech_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Footer
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,
        textColor=colors.grey
    )
    story.append(Paragraph("Generated by Deepfake Audio Detection System v1.0", footer_style))
    
    # Build PDF
    doc.build(story)
    
    return filename

@app.route('/')
def index():
    """Main landing page."""
    # Redirect to analyze if already logged in
    if 'logged_in' in session:
        return redirect(url_for('analyze'))
    return render_template('main.html')

@app.route('/analyze')
@login_required
def analyze():
    """Analysis page with React interface."""
    return render_template('react_app.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt - Username: {username}")
        
        # Query user from database
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password) and user.is_active:
            session.clear()
            session['logged_in'] = True
            session['username'] = user.username
            session['user_id'] = user.id
            session['user_role'] = user.role
            session.permanent = True
            
            # Update last login
            user.update_last_login()
            
            print(f"Login successful for user: {username}")
            return redirect(url_for('analyze'))
        else:
            print(f"Login failed for user: {username}")
            return render_template('login.html', error='Invalid username or password')
    
    # Check if already logged in
    if 'logged_in' in session:
        print(f"User {session.get('username')} already logged in, redirecting to analyze")
        return redirect(url_for('analyze'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register new user."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        # Validation
        if not all([username, email, full_name, password, password_confirm]):
            return render_template('register.html', error='Tüm alanları doldurun')
        
        if password != password_confirm:
            return render_template('register.html', error='Şifreler eşleşmiyor')
        
        if len(password) < 6:
            return render_template('register.html', error='Şifre en az 6 karakter olmalı')
        
        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error='Bu kullanıcı adı zaten kullanılıyor')
        
        # Check if email already exists
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return render_template('register.html', error='Bu e-posta adresi zaten kayıtlı')
        
        try:
            # Create new user
            new_user = User(
                username=username,
                email=email,
                full_name=full_name,
                role='user'
            )
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            print(f"New user registered: {username}")
            return render_template('register.html', success='Hesabınız başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.')
        
        except Exception as e:
            db.session.rollback()
            print(f"Error registering user: {e}")
            return render_template('register.html', error='Kayıt sırasında bir hata oluştu')
    
    # Check if already logged in
    if 'logged_in' in session:
        return redirect(url_for('analyze'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user."""
    # Clean up uploaded file before logout
    if 'last_uploaded_file' in session:
        old_filepath = session['last_uploaded_file']
        if os.path.exists(old_filepath):
            try:
                os.remove(old_filepath)
                print(f"Cleaned up file on logout: {old_filepath}")
            except Exception as e:
                print(f"Could not delete file on logout: {e}")
    
    session.clear()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload and process audio."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload WAV, MP3, FLAC, M4A, OGG, or WEBM files.'}), 400
    
    try:
        # Delete previous uploaded file if exists
        if 'last_uploaded_file' in session:
            old_filepath = session['last_uploaded_file']
            if os.path.exists(old_filepath):
                try:
                    os.remove(old_filepath)
                    print(f"Deleted old file: {old_filepath}")
                except Exception as e:
                    print(f"Could not delete old file: {e}")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Store filepath in session for later cleanup
        session['last_uploaded_file'] = filepath
        
        print(f"File saved: {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        
        # Extract features
        spectrogram, audio, sr, original_duration = extract_audio_features(filepath)
        if spectrogram is None:
            return jsonify({'error': 'Error processing audio file'}), 500
        
        # Predict
        result = predict_deepfake(spectrogram)
        if result is None:
            return jsonify({'error': 'Error during prediction'}), 500
        
        # Create visualization
        plot_data = create_spectrogram_plot(spectrogram, audio, sr)
        
        # Prepare response
        response_data = {
            'file_id': file_id,
            'filename': filename,
            'is_fake': result['is_fake'],
            'confidence': result['confidence'],
            'raw_probability': result['raw_probability'],
            'threshold_used': result['threshold_used'],
            'plot_data': plot_data,
            'file_size': os.path.getsize(filepath),
            'duration': original_duration,  # Use original duration, not processed
            'file_path': f'/uploads/{unique_filename}'  # Use unique_filename, not original filename
        }
        
        print(f"Audio file path for playback: {response_data['file_path']}")
        print(f"Actual file on disk: {filepath}")
        print(f"File still exists: {os.path.exists(filepath)}")
        
        # Save analysis to database
        try:
            analysis = AnalysisHistory(
                user_id=session.get('user_id'),
                filename=filename,
                file_size=response_data['file_size'],
                duration=response_data['duration'],
                is_fake=result['is_fake'],
                confidence=result['confidence'],
                raw_probability=result['raw_probability'],
                threshold_used=result['threshold_used']
            )
            db.session.add(analysis)
            db.session.commit()
            print(f"Analysis saved to database for user {session.get('username')}")
        except Exception as db_error:
            print(f"Error saving to database: {db_error}")
            db.session.rollback()
        
        # Don't clean up uploaded file - keep it for playback
        # File will be accessible via /uploads/<filename> route
        # os.remove(filepath)  # Commented out to allow audio playback
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    """Generate and download TXT report."""
    try:
        data = request.json
        print(f"Generating report for: {data.get('filename')}")
        
        # Generate TXT report
        filename = f"deepfake_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create report content
        result_text = "REAL AUDIO" if not data['is_fake'] else "DEEPFAKE AUDIO"
        report_content = f"""
╔══════════════════════════════════════════════════════════════╗
║        DEEPFAKE AUDIO DETECTION REPORT                       ║
╚══════════════════════════════════════════════════════════════╝

Report Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Analysis Model: Advanced CNN (90% Accuracy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File Name:        {data['filename']}
File Size:        {data.get('file_size', 0) / 1024:.2f} KB
Duration:         {data.get('duration', 0):.2f} seconds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classification:   *** {result_text} ***
Confidence:       {data['confidence']:.1f}%
Raw Probability:  {data['raw_probability']:.6f}
Threshold Used:   {data['threshold_used']:.6f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Type:       Convolutional Neural Network (CNN)
Model Accuracy:   90.0%
F1 Score:         89.8%
Feature Type:     Mel-Spectrogram

Analysis Method:
The audio file was processed using advanced machine learning
techniques. The system extracts mel-spectrogram features and
analyzes them using a trained CNN model.

Confidence Interpretation:
The confidence level indicates how certain the model is about
its prediction. Higher confidence values suggest more reliable
results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This analysis is based on state-of-the-art deepfake detection
technology. For critical applications, we recommend additional
verification methods.

Generated by DeepFake Guard v1.0
User: {session.get('username')}
Report ID: {filename.replace('.txt', '')}
"""
        
        # Write report to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Report generated: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'report_url': f'/download_report/{filename}'
        })
    
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Download generated PDF report."""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/uploads/<filename>')
@login_required
def serve_upload(filename):
    """Serve uploaded audio files for playback."""
    try:
        # Get file extension to determine MIME type
        ext = filename.rsplit('.', 1)[-1].lower()
        
        # MIME type mapping for audio files
        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4',
            'ogg': 'audio/ogg',
            'webm': 'audio/webm'
        }
        
        mimetype = mime_types.get(ext, 'audio/mpeg')
        
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename,
            mimetype=mimetype,
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/history')
@login_required
def get_history():
    """Get user's analysis history."""
    try:
        user_id = session.get('user_id')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Query user's analysis history with pagination
        pagination = AnalysisHistory.query.filter_by(user_id=user_id)\
            .order_by(AnalysisHistory.analyzed_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        history_data = []
        for analysis in pagination.items:
            history_data.append({
                'id': analysis.id,
                'filename': analysis.filename,
                'is_fake': analysis.is_fake,
                'confidence': analysis.confidence,
                'file_size': analysis.file_size,
                'duration': analysis.duration,
                'analyzed_at': analysis.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({
            'history': history_data,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page
        })
    
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
@login_required
def get_stats():
    """Get user's analysis statistics."""
    try:
        user_id = session.get('user_id')
        
        total_analyses = AnalysisHistory.query.filter_by(user_id=user_id).count()
        fake_count = AnalysisHistory.query.filter_by(user_id=user_id, is_fake=True).count()
        real_count = total_analyses - fake_count
        
        # Get average confidence
        analyses = AnalysisHistory.query.filter_by(user_id=user_id).all()
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses) if analyses else 0
        
        return jsonify({
            'total_analyses': total_analyses,
            'fake_count': fake_count,
            'real_count': real_count,
            'avg_confidence': round(avg_confidence, 2)
        })
    
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Error downloading report: {e}")
        return "Report not found", 404

if __name__ == '__main__':
    print("Starting Deepfake Audio Detection Web Interface...")
    
    # Load model and scaler
    if load_model_and_scaler():
        print("System ready!")
        print("Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("Failed to load model. Please check the model files.")
