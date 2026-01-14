// JavaScript for Deepfake Audio Detection Interface

let currentFileData = null;
let currentResult = null;

// DOM elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const uploadSection = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const downloadReportBtn = document.getElementById('download-report-btn');
const analyzeAnotherBtn = document.getElementById('analyze-another-btn');
const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // File upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Button events
    analyzeBtn.addEventListener('click', analyzeAudio);
    downloadReportBtn.addEventListener('click', generateReport);
    analyzeAnotherBtn.addEventListener('click', resetInterface);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/mp4', 'audio/ogg'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg'];
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        showError('Invalid file format. Please upload WAV, MP3, FLAC, M4A, or OGG files.');
        return;
    }
    
    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Please upload files smaller than 16MB.');
        return;
    }
    
    // Store file data
    currentFileData = file;
    
    // Update UI
    updateUploadDisplay(file);
    analyzeBtn.disabled = false;
}

function updateUploadDisplay(file) {
    const uploadArea = document.getElementById('upload-area');
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
    
    uploadArea.innerHTML = `
        <i class="fas fa-file-audio upload-icon text-success"></i>
        <h4 class="text-success">File Ready</h4>
        <p class="text-muted">${file.name}</p>
        <small class="text-muted">Size: ${sizeInMB} MB</small>
    `;
}

async function analyzeAudio() {
    if (!currentFileData) {
        showError('Please select a file first.');
        return;
    }
    
    try {
        // Show loading
        showSection('loading');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFileData);
        
        // Send request
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Store result and display
        currentResult = result;
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
        showSection('upload');
    }
}

function displayResults(result) {
    // Update main result
    const resultBox = document.querySelector('.result-box');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const confidenceText = document.getElementById('confidence-text');
    const resultHeader = document.getElementById('result-header');
    
    if (result.is_fake) {
        resultBox.className = 'result-box fake';
        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        resultText.textContent = 'DEEPFAKE DETECTED';
        resultHeader.className = 'card-header bg-danger text-white';
    } else {
        resultBox.className = 'result-box real';
        resultIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
        resultText.textContent = 'REAL AUDIO';
        resultHeader.className = 'card-header bg-success text-white';
    }
    
    confidenceText.textContent = `Confidence: ${result.confidence.toFixed(1)}%`;
    
    // Update file information
    document.getElementById('file-name').textContent = result.filename;
    document.getElementById('file-duration').textContent = `${result.duration.toFixed(2)} seconds`;
    document.getElementById('file-size').textContent = `${(result.file_size / 1024).toFixed(1)} KB`;
    document.getElementById('raw-probability').textContent = result.raw_probability.toFixed(6);
    document.getElementById('threshold-used').textContent = result.threshold_used.toFixed(6);
    
    // Update spectrogram plot
    const spectrogramPlot = document.getElementById('spectrogram-plot');
    spectrogramPlot.src = `data:image/png;base64,${result.plot_data}`;
    
    // Show results section with animation
    showSection('results');
    
    // Add animation classes
    setTimeout(() => {
        resultsSection.classList.add('fade-in');
    }, 100);
}

async function generateReport() {
    if (!currentResult) {
        showError('No analysis results available.');
        return;
    }
    
    try {
        // Show loading state
        downloadReportBtn.disabled = true;
        downloadReportBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating Report...';
        
        // Prepare report data
        const reportData = {
            filename: currentResult.filename,
            is_fake: currentResult.is_fake,
            confidence: currentResult.confidence,
            raw_probability: currentResult.raw_probability,
            threshold_used: currentResult.threshold_used,
            file_size: currentResult.file_size,
            duration: currentResult.duration
        };
        
        // Send request to generate report
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(reportData)
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Download the report using a temporary link
        const link = document.createElement('a');
        link.href = result.report_url;
        link.download = ''; // Browser will use filename from Content-Disposition header
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show success message
        showSuccess('Report downloaded successfully!');
        
    } catch (error) {
        console.error('Report generation error:', error);
        showError(`Failed to generate report: ${error.message}`);
    } finally {
        // Reset button state
        downloadReportBtn.disabled = false;
        downloadReportBtn.innerHTML = '<i class="fas fa-download me-2"></i>Download Report (TXT)';
    }
}

function resetInterface() {
    // Clear data
    currentFileData = null;
    currentResult = null;
    
    // Reset file input
    fileInput.value = '';
    
    // Reset upload area
    uploadArea.innerHTML = `
        <i class="fas fa-cloud-upload-alt upload-icon"></i>
        <h4>Drag & Drop Audio File Here</h4>
        <p class="text-muted">or click to browse</p>
    `;
    
    // Reset button
    analyzeBtn.disabled = true;
    
    // Show upload section
    showSection('upload');
}

function showSection(sectionName) {
    // Hide all sections
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Show requested section
    switch (sectionName) {
        case 'upload':
            uploadSection.style.display = 'block';
            break;
        case 'loading':
            loadingSection.style.display = 'block';
            break;
        case 'results':
            resultsSection.style.display = 'block';
            break;
    }
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(message) {
    document.getElementById('error-message').textContent = message;
    errorModal.show();
}

function showSuccess(message) {
    // Create and show success toast
    const toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
    toastContainer.style.zIndex = '9999';
    
    toastContainer.innerHTML = `
        <div class="toast show" role="alert">
            <div class="toast-header bg-success text-white">
                <i class="fas fa-check-circle me-2"></i>
                <strong class="me-auto">Success</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    document.body.appendChild(toastContainer);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toastContainer.parentNode) {
            toastContainer.parentNode.removeChild(toastContainer);
        }
    }, 5000);
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Add some visual feedback for buttons
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('btn')) {
        e.target.style.transform = 'scale(0.95)';
        setTimeout(() => {
            e.target.style.transform = '';
        }, 150);
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Escape key to reset
    if (e.key === 'Escape') {
        resetInterface();
    }
    
    // Enter key to analyze (if file is selected)
    if (e.key === 'Enter' && currentFileData && !analyzeBtn.disabled) {
        analyzeAudio();
    }
});

// Add loading progress simulation
function simulateProgress() {
    const progressBar = document.querySelector('.progress-bar');
    let width = 0;
    const interval = setInterval(() => {
        width += Math.random() * 10;
        if (width >= 100) {
            width = 100;
            clearInterval(interval);
        }
        progressBar.style.width = width + '%';
    }, 200);
}
