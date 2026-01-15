// JavaScript for Deepfake Audio Detection Interface

let currentFileData = null;
let currentResult = null;
let recordedAudioURL = null; // Store recorded audio URL for download
let currentAudioPath = null; // Store current audio path for playback

// Microphone recording variables
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = null;
let timerInterval = null;

// DOM elements (will be initialized after DOMContentLoaded)
let uploadArea, fileInput, analyzeBtn, uploadSection, loadingSection, resultsSection;
let downloadReportBtn, downloadAudioBtn, playAudioBtn, audioPlayer, analyzeAnotherBtn, errorModal;
let recordBtn, stopRecordBtn, recordingTimer, timerDisplay, recordingStatus;

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize DOM elements
    uploadArea = document.getElementById('upload-area');
    fileInput = document.getElementById('file-input');
    analyzeBtn = document.getElementById('analyze-btn');
    uploadSection = document.getElementById('upload-section');
    loadingSection = document.getElementById('loading-section');
    resultsSection = document.getElementById('results-section');
    downloadReportBtn = document.getElementById('download-report-btn');
    downloadAudioBtn = document.getElementById('download-audio-btn');
    playAudioBtn = document.getElementById('play-audio-btn');
    audioPlayer = document.getElementById('audio-player');
    analyzeAnotherBtn = document.getElementById('analyze-another-btn');
    errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
    
    // Recording elements
    recordBtn = document.getElementById('record-btn');
    stopRecordBtn = document.getElementById('stop-record-btn');
    recordingTimer = document.getElementById('recording-timer');
    timerDisplay = document.getElementById('timer-display');
    recordingStatus = document.getElementById('recording-status');
    
    console.log('Recording button found:', recordBtn);
    
    initializeEventListeners();
});

function initializeEventListeners() {
    // File upload area events
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }
    
    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Button events
    if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeAudio);
    if (downloadReportBtn) downloadReportBtn.addEventListener('click', generateReport);
    if (downloadAudioBtn) downloadAudioBtn.addEventListener('click', downloadRecordedAudio);
    if (playAudioBtn) playAudioBtn.addEventListener('click', playAudio);
    if (analyzeAnotherBtn) analyzeAnotherBtn.addEventListener('click', resetInterface);
    
    // Recording events
    if (recordBtn) {
        recordBtn.addEventListener('click', startRecording);
        console.log('Record button event listener added');
    } else {
        console.error('Record button not found!');
    }
    
    if (stopRecordBtn) {
        stopRecordBtn.addEventListener('click', stopRecording);
    }
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
    
    // Clear any recorded audio when a new file is selected
    if (recordedAudioURL) {
        URL.revokeObjectURL(recordedAudioURL);
        recordedAudioURL = null;
        console.log('Cleared recorded audio URL');
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
    
    // Set audio source for playback
    console.log('Setting up audio player...');
    console.log('recordedAudioURL:', recordedAudioURL);
    console.log('result.file_path:', result.file_path);
    console.log('currentFileData type:', currentFileData?.name);
    
    // Prefer blob URL for recorded audio, otherwise use server file path
    if (recordedAudioURL) {
        // Use blob URL for recorded audio (works better across browsers)
        currentAudioPath = recordedAudioURL;
        downloadAudioBtn.style.display = 'inline-block';
        console.log('Using recorded audio blob URL');
    } else if (result.file_path) {
        // Use uploaded file path from server
        currentAudioPath = result.file_path;
        downloadAudioBtn.style.display = 'none';
        console.log('Using uploaded file path:', currentAudioPath);
    }
    
    // Configure audio player
    if (currentAudioPath) {
        console.log('Setting audio player source to:', currentAudioPath);
        
        // Wait a bit before setting audio source to ensure file is ready
        setTimeout(() => {
            // Clear existing sources
            audioPlayer.innerHTML = '';
            
            // Determine audio type from path
            let audioType = 'audio/mpeg'; // default
            
            // Check if it's a blob URL (recorded audio)
            if (currentAudioPath.startsWith('blob:')) {
                // For recorded audio, browser will auto-detect the format
                console.log('Using blob URL (recorded audio):', currentAudioPath);
                
                // Clear any existing source elements
                audioPlayer.innerHTML = '';
                
                // Remove type attribute - let browser auto-detect
                audioPlayer.removeAttribute('type');
                
                // Set preload to auto for better compatibility
                audioPlayer.preload = 'auto';
                
                // Set the blob URL as source
                audioPlayer.src = currentAudioPath;
                
                console.log('Audio player src set to blob URL');
                console.log('Audio player ready state:', audioPlayer.readyState);
                
                // Add success event listeners
                audioPlayer.addEventListener('loadedmetadata', () => {
                    console.log('✅ Blob audio metadata loaded, duration:', audioPlayer.duration);
                }, { once: true });
                
                audioPlayer.addEventListener('loadeddata', () => {
                    console.log('✅ Blob audio data loaded and ready to play');
                    playAudioBtn.disabled = false;
                }, { once: true });
                
                audioPlayer.addEventListener('canplay', () => {
                    console.log('✅ Blob audio can play now');
                }, { once: true });
                
                // Trigger load
                console.log('Calling audioPlayer.load()...');
                audioPlayer.load();
            } else {
                // For file paths, detect type from extension
                if (currentAudioPath.includes('.wav')) {
                    audioType = 'audio/wav';
                } else if (currentAudioPath.includes('.mp3')) {
                    audioType = 'audio/mpeg';
                } else if (currentAudioPath.includes('.flac')) {
                    audioType = 'audio/flac';
                } else if (currentAudioPath.includes('.m4a')) {
                    audioType = 'audio/mp4';
                } else if (currentAudioPath.includes('.ogg')) {
                    audioType = 'audio/ogg';
                }
                
                console.log('Audio type detected:', audioType);
                
                // Create source element
                const source = document.createElement('source');
                source.src = currentAudioPath;
                source.type = audioType;
                audioPlayer.appendChild(source);
                
                // Also set src directly as fallback
                audioPlayer.src = currentAudioPath;
            }
            
            // Add error listener before loading
            audioPlayer.onerror = (e) => {
                console.error('Audio player error:', e);
                console.error('Audio player error code:', audioPlayer.error?.code);
                console.error('Audio player error message:', audioPlayer.error?.message);
                console.error('Audio player network state:', audioPlayer.networkState);
                console.error('Audio player ready state:', audioPlayer.readyState);
                console.error('Attempted to load:', currentAudioPath);
                
                let errorMsg = 'Failed to load audio file';
                if (audioPlayer.error) {
                    switch(audioPlayer.error.code) {
                        case 1: errorMsg = 'Audio loading aborted'; break;
                        case 2: errorMsg = 'Network error while loading audio'; break;
                        case 3: errorMsg = 'Audio format not supported'; break;
                        case 4: errorMsg = 'Audio source not found - File may have been deleted'; break;
                    }
                }
                showError(errorMsg);
            };
            
            audioPlayer.onloadedmetadata = () => {
                console.log('Audio metadata loaded');
                console.log('Duration:', audioPlayer.duration);
            };
            
            audioPlayer.onloadeddata = () => {
                console.log('Audio data loaded successfully');
                playAudioBtn.disabled = false;
            };
            
            audioPlayer.oncanplay = () => {
                console.log('Audio can play');
            };
            
            // Force reload
            audioPlayer.load();
        }, 500); // Wait 500ms for file to be ready
        
    } else {
        console.warn('No audio path available');
    }
    
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

function downloadRecordedAudio() {
    if (!recordedAudioURL) {
        showError('No recorded audio available.');
        return;
    }
    
    try {
        // Create download link
        const link = document.createElement('a');
        link.href = recordedAudioURL;
        link.download = `recorded_audio_${Date.now()}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showSuccess('Audio downloaded successfully!');
    } catch (error) {
        console.error('Audio download error:', error);
        showError('Failed to download audio.');
    }
}

function playAudio() {
    console.log('playAudio called');
    console.log('currentAudioPath:', currentAudioPath);
    console.log('audioPlayer.src:', audioPlayer.src);
    console.log('audioPlayer.readyState:', audioPlayer.readyState);
    
    if (!currentAudioPath) {
        showError('No audio available to play.');
        return;
    }
    
    try {
        if (audioPlayer.paused) {
            console.log('Attempting to play audio...');
            const playPromise = audioPlayer.play();
            
            if (playPromise !== undefined) {
                playPromise
                    .then(() => {
                        console.log('Audio playing successfully');
                        playAudioBtn.innerHTML = '<i class="fas fa-pause me-2"></i>Pause Audio';
                        
                        // Reset button when audio ends
                        audioPlayer.onended = () => {
                            playAudioBtn.innerHTML = '<i class="fas fa-play me-2"></i>Play Audio';
                        };
                    })
                    .catch(error => {
                        console.error('Play failed:', error);
                        showError('Failed to play audio: ' + error.message);
                    });
            }
        } else {
            console.log('Pausing audio...');
            audioPlayer.pause();
            playAudioBtn.innerHTML = '<i class="fas fa-play me-2"></i>Play Audio';
        }
    } catch (error) {
        console.error('Audio playback error:', error);
        showError('Failed to play audio.');
    }
}

function resetInterface() {
    // Clear data
    currentFileData = null;
    
    // Revoke blob URL to free memory
    if (recordedAudioURL) {
        URL.revokeObjectURL(recordedAudioURL);
    }
    recordedAudioURL = null;
    
    currentResult = null;
    currentAudioPath = null;
    
    // Reset audio player
    if (audioPlayer) {
        // Remove all event listeners to prevent errors
        audioPlayer.onerror = null;
        audioPlayer.onloadeddata = null;
        audioPlayer.onloadedmetadata = null;
        audioPlayer.oncanplay = null;
        audioPlayer.onended = null;
        
        // Pause and clear
        audioPlayer.pause();
        audioPlayer.removeAttribute('src');
        audioPlayer.load(); // This clears the player
    }
    
    // Reset play button
    if (playAudioBtn) {
        playAudioBtn.innerHTML = '<i class="fas fa-play me-2"></i>Play Audio';
        playAudioBtn.disabled = true;
    }
    
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
    
    // Reset recording status message
    if (recordingStatus) {
        recordingStatus.textContent = '';
        recordingStatus.classList.remove('text-success', 'text-danger');
    }
    
    // Reset recording buttons to default state
    if (recordBtn) recordBtn.style.display = 'inline-block';
    if (stopRecordBtn) stopRecordBtn.style.display = 'none';
    if (recordingTimer) recordingTimer.style.display = 'none';
    
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

// ===== MICROPHONE RECORDING FUNCTIONS =====

async function startRecording() {
    console.log('startRecording called');
    
    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Your browser does not support microphone recording. Please use a modern browser.');
        console.error('MediaDevices API not supported');
        return;
    }
    
    try {
        console.log('Requesting microphone access...');
        
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 44100
            } 
        });
        
        console.log('Microphone access granted');
        
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        console.log('MediaRecorder created');
        
        // Handle data available
        mediaRecorder.ondataavailable = (event) => {
            console.log('Data available:', event.data.size, 'bytes');
            audioChunks.push(event.data);
        };
        
        // Handle recording stop
        mediaRecorder.onstop = async () => {
            console.log('Recording stopped, processing audio...');
            
            // Get the MIME type that was actually used by MediaRecorder
            const mimeType = mediaRecorder.mimeType || 'audio/webm';
            console.log('MediaRecorder MIME type:', mimeType);
            
            // Create audio blob with the correct MIME type
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            console.log('Audio blob created:', audioBlob.size, 'bytes', 'type:', audioBlob.type);
            
            // Create downloadable URL for the recorded audio
            recordedAudioURL = URL.createObjectURL(audioBlob);
            console.log('Audio URL created for download:', recordedAudioURL);
            
            // Determine file extension based on MIME type
            let extension = 'webm';
            if (mimeType.includes('ogg')) {
                extension = 'ogg';
            } else if (mimeType.includes('mp4')) {
                extension = 'm4a';
            } else if (mimeType.includes('wav')) {
                extension = 'wav';
            }
            
            // Convert to file with correct MIME type
            const audioFile = new File([audioBlob], `recording_${Date.now()}.${extension}`, {
                type: mimeType
            });
            
            console.log('Audio file created:', audioFile.name, 'type:', audioFile.type);
            
            // Set as current file
            currentFileData = audioFile;
            
            // Update UI
            updateUploadAreaWithRecording(audioFile);
            analyzeBtn.disabled = false;
            
            recordingStatus.textContent = '✅ Recording completed! Click analyze button.';
            recordingStatus.classList.remove('text-danger');
            recordingStatus.classList.add('text-success');
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            console.log('All tracks stopped');
        };
        
        // Start recording
        mediaRecorder.start();
        recordingStartTime = Date.now();
        console.log('Recording started');
        
        // Update UI
        recordBtn.style.display = 'none';
        stopRecordBtn.style.display = 'inline-block';
        recordingTimer.style.display = 'block';
        recordingStatus.textContent = '🎤 Recording...';
        recordingStatus.classList.remove('text-success');
        recordingStatus.classList.add('text-danger');
        
        // Start timer
        timerInterval = setInterval(updateTimer, 100);
        
        // Auto-stop after 10 seconds (max duration)
        setTimeout(() => {
            console.log('Auto-stop timeout reached (10 seconds)');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                console.log('Stopping recording automatically after 10 seconds');
                stopRecording();
            }
        }, 10000);
        
    } catch (error) {
        console.error('Microphone access error:', error);
        console.error('Error details:', error.name, error.message);
        
        let errorMessage = 'Microphone access denied. ';
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage += 'Please grant microphone permission in browser settings.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found. Please connect a microphone.';
        } else {
            errorMessage += error.message;
        }
        
        showError(errorMessage);
    }
}

function stopRecording() {
    console.log('stopRecording called');
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        console.log('Stopping MediaRecorder...');
        mediaRecorder.stop();
        
        // Stop timer
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        
        // Update UI
        recordBtn.style.display = 'inline-block';
        stopRecordBtn.style.display = 'none';
        recordingTimer.style.display = 'none';
    }
}

function updateTimer() {
    if (recordingStartTime) {
        const elapsed = Date.now() - recordingStartTime;
        const seconds = Math.floor(elapsed / 1000);
        const milliseconds = Math.floor((elapsed % 1000) / 100);
        timerDisplay.textContent = `${seconds}:${milliseconds}`;
    }
}

function updateUploadAreaWithRecording(file) {
    uploadArea.innerHTML = `
        <i class="fas fa-microphone text-danger" style="font-size: 3rem;"></i>
        <h4 class="mt-3">Recording Completed!</h4>
        <p class="text-muted">${file.name}</p>
        <p class="text-muted small">Size: ${(file.size / 1024).toFixed(2)} KB</p>
    `;
}

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
