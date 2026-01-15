import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [recording, setRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [recordedAudioURL, setRecordedAudioURL] = useState(null)
  
  const fileInputRef = useRef(null)
  const audioPlayerRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const timerIntervalRef = useRef(null)

  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current)
      }
    }
  }, [])

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      validateAndSetFile(selectedFile)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      validateAndSetFile(droppedFile)
    }
  }

  const validateAndSetFile = (selectedFile) => {
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/mp4', 'audio/ogg', 'audio/webm']
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase()
    const allowedExtensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm']
    
    if (!allowedTypes.includes(selectedFile.type) && !allowedExtensions.includes(fileExtension)) {
      setError('Invalid file format. Please upload WAV, MP3, FLAC, M4A, OGG, or WEBM files.')
      return
    }
    
    if (selectedFile.size > 16 * 1024 * 1024) {
      setError('File size too large. Please upload files smaller than 16MB.')
      return
    }
    
    if (recordedAudioURL) {
      URL.revokeObjectURL(recordedAudioURL)
      setRecordedAudioURL(null)
    }
    
    setFile(selectedFile)
    setError(null)
    setResult(null)
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }
      
      mediaRecorderRef.current.onstop = async () => {
        const mimeType = mediaRecorderRef.current.mimeType || 'audio/webm'
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType })
        const url = URL.createObjectURL(audioBlob)
        setRecordedAudioURL(url)
        
        const extension = mimeType.includes('ogg') ? 'ogg' : 
                         mimeType.includes('mp4') ? 'm4a' :
                         mimeType.includes('wav') ? 'wav' : 'webm'
        
        const audioFile = new File([audioBlob], `recording_${Date.now()}.${extension}`, { type: mimeType })
        setFile(audioFile)
        setError(null)
        setResult(null)
        
        stream.getTracks().forEach(track => track.stop())
      }
      
      mediaRecorderRef.current.start()
      setRecording(true)
      setRecordingTime(0)
      
      timerIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
      
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current)
      }
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first.')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setRecordedAudioURL(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleDownloadReport = async () => {
    if (!result) return
    
    try {
      const response = await axios.post('/generate_report', {
        filename: result.filename,
        is_fake: result.is_fake,
        confidence: result.confidence,
        raw_probability: result.raw_probability,
        threshold_used: result.threshold_used,
        file_size: result.file_size || 0,
        duration: result.duration || 0
      })
      
      // Download the file directly
      window.location.href = `/download_report/${response.data.filename}`
    } catch (err) {
      setError('Failed to generate report')
    }
  }

  const handleDownloadRecording = () => {
    if (recordedAudioURL) {
      const link = document.createElement('a')
      link.href = recordedAudioURL
      link.download = file?.name || 'recording.webm'
      link.click()
    }
  }

  const playAudio = () => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.play()
    }
  }

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
        <div className="container">
          <a className="navbar-brand" href="/" style={{display: 'flex', alignItems: 'center'}}>
            <img src="/static/logo.png" alt="Logo" style={{height: '45px', marginRight: '10px', filter: 'brightness(0) invert(1)'}} />
            DeepFakeGuard
          </a>
          <div className="navbar-nav ms-auto">
            <a href="/logout" className="btn btn-outline-light btn-sm">
              <i className="fas fa-sign-out-alt me-1"></i>
              Logout
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="hero-section">
        <div className="container">
          <div className="row justify-content-center">
            <div className="col-lg-8 text-center">
              <h1 className="display-4 fw-bold mb-4">
                Deepfake Speech Detection
              </h1>
              <p className="lead mb-5">
                Upload your Speech file and let our AI-powered system analyze whether it's bona fide or artificially generated.
                Using state-of-the-art CNN technology.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="container my-5">
        {/* Upload Section */}
        {!loading && !result && (
          <div className="row justify-content-center">
            <div className="col-lg-8">
              <div className="card shadow-lg border-0">
                <div className="card-header bg-primary text-white">
                  <h3 className="card-title mb-0">
                    <i className="fas fa-upload me-2"></i>
                    Upload Speech File
                  </h3>
                </div>
                <div className="card-body p-4">
                  <div 
                    className="upload-area" 
                    onClick={() => fileInputRef.current?.click()}
                    onDrop={handleDrop}
                    onDragOver={(e) => e.preventDefault()}
                  >
                    {!file ? (
                      <>
                        <i className="fas fa-cloud-upload-alt upload-icon"></i>
                        <h4>Drag & Drop Speech File Here</h4>
                        <p className="text-muted">or click to browse</p>
                      </>
                    ) : (
                      <>
                        <i className="fas fa-file-audio upload-icon text-success"></i>
                        <h4 className="text-success">File Ready</h4>
                        <p className="text-muted">{file.name}</p>
                        <small className="text-muted">Size: {(file.size / (1024 * 1024)).toFixed(2)} MB</small>
                      </>
                    )}
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileSelect}
                      accept=".wav,.mp3,.flac,.m4a,.ogg,.webm"
                      style={{ display: 'none' }}
                    />
                  </div>
                  
                  <div className="text-center my-3">
                    <div className="d-inline-block px-3 py-2 bg-light rounded">
                      <span className="text-muted fw-bold">OR</span>
                    </div>
                  </div>

                  {/* Microphone Recording */}
                  <div className="record-section p-3 border rounded" style={{ background: 'linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%)' }}>
                    <div className="alert alert-warning small mb-3" role="alert">
                      <i className="fas fa-exclamation-triangle me-2"></i>
                      <strong>Note:</strong> Recorded audio may not work on all browsers. For best results, upload an audio file directly.
                    </div>
                    <div className="text-center">
                      <i className="fas fa-microphone fa-3x text-danger mb-3"></i>
                      <h5>Record from Microphone</h5>
                      <p className="text-muted small mb-3">Record audio and analyze directly</p>
                      
                      {!recording ? (
                        <button onClick={startRecording} className="btn btn-danger btn-lg">
                          <i className="fas fa-microphone me-2"></i>
                          Start Recording
                        </button>
                      ) : (
                        <button onClick={stopRecording} className="btn btn-secondary btn-lg">
                          <i className="fas fa-stop me-2"></i>
                          Stop Recording
                        </button>
                      )}
                      
                      {recording && (
                        <div className="mt-2 fw-bold text-danger" style={{ fontSize: '1.2rem' }}>
                          <i className="fas fa-circle fa-beat" style={{ fontSize: '0.8rem' }}></i>
                          <span className="ms-2">{formatTime(recordingTime)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <small className="text-muted">
                      <i className="fas fa-info-circle me-1"></i>
                      Supported formats: WAV, MP3, FLAC, M4A, OGG, WEBM • Max size: 16MB
                    </small>
                  </div>
                  
                  {error && (
                    <div className="alert alert-danger mt-3" role="alert">
                      <i className="fas fa-exclamation-circle me-2"></i>
                      {error}
                    </div>
                  )}
                  
                  <div className="mt-4">
                    <button 
                      onClick={handleAnalyze} 
                      className="btn btn-success btn-lg w-100" 
                      disabled={!file}
                    >
                      <i className="fas fa-search me-2"></i>
                      Analyze Speech
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading Section */}
        {loading && (
          <div className="row justify-content-center">
            <div className="col-lg-8">
              <div className="card shadow-lg border-0">
                <div className="card-body text-center p-5">
                  <div className="spinner-border text-primary mb-3" role="status" style={{ width: '3rem', height: '3rem' }}>
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <h4>Analyzing Speech...</h4>
                  <p className="text-muted">Our AI is processing your file. This may take a few seconds.</p>
                  <div className="progress mt-3">
                    <div className="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style={{ width: '100%' }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="row justify-content-center">
            <div className="col-lg-10">
              <div className="card shadow-lg border-0 mb-4">
                <div className={`card-header ${result.is_fake ? 'bg-danger' : 'bg-success'} text-white`}>
                  <h3 className="card-title mb-0">
                    <i className="fas fa-chart-line me-2"></i>
                    Detection Results
                  </h3>
                </div>
                <div className="card-body p-4">
                  <div className="row">
                    <div className="col-md-6">
                      <div className={`result-box ${result.is_fake ? 'fake' : 'real'}`}>
                        <div className="result-icon">
                          <i className={`fas ${result.is_fake ? 'fa-exclamation-triangle' : 'fa-check-circle'}`}></i>
                        </div>
                        <h2>{result.is_fake ? 'DEEPFAKE DETECTED' : 'REAL AUDIO'}</h2>
                        <p className="confidence">Confidence: {result.confidence.toFixed(1)}%</p>
                      </div>
                    </div>
                    <div className="col-md-6">
                      <div className="file-info">
                        <h5><i className="fas fa-file-audio me-2"></i>File Information</h5>
                        <table className="table table-sm">
                          <tbody>
                            <tr>
                              <td><strong>Filename:</strong></td>
                              <td>{result.filename}</td>
                            </tr>
                            <tr>
                              <td><strong>Duration:</strong></td>
                              <td>{result.duration.toFixed(2)} seconds</td>
                            </tr>
                            <tr>
                              <td><strong>Size:</strong></td>
                              <td>{(result.file_size / 1024).toFixed(1)} KB</td>
                            </tr>
                            <tr>
                              <td><strong>Probability:</strong></td>
                              <td>{result.raw_probability.toFixed(6)}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <h5><i className="fas fa-cogs me-2"></i>Technical Analysis</h5>
                    <div className="row">
                      <div className="col-md-6">
                        <div className="tech-detail">
                          <span className="badge bg-info">Model Used</span>
                          <span>Advanced CNN</span>
                        </div>
                        <div className="tech-detail">
                          <span className="badge bg-info">Threshold</span>
                          <span>{result.threshold_used.toFixed(6)}</span>
                        </div>
                      </div>
                      <div className="col-md-6">
                        <div className="tech-detail">
                          <span className="badge bg-info">Analysis Method</span>
                          <span>Mel-Spectrogram CNN</span>
                        </div>
                        <div className="tech-detail">
                          <span className="badge bg-info">Model F1-Score</span>
                          <span>89.8%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {result.plot_data && (
                    <div className="mt-4">
                      <h5><i className="fas fa-chart-area me-2"></i>Speech Visualization</h5>
                      <div className="text-center">
                        <img 
                          src={`data:image/png;base64,${result.plot_data}`} 
                          alt="Speech Analysis" 
                          className="img-fluid rounded shadow"
                        />
                      </div>
                    </div>
                  )}

                  {(result.file_path || recordedAudioURL) && (
                    <div className="mt-4 text-center">
                      <audio 
                        ref={audioPlayerRef}
                        controls 
                        preload="metadata" 
                        style={{ maxWidth: '100%', width: '500px' }}
                        src={recordedAudioURL || result.file_path}
                      >
                        Your browser does not support the audio element.
                      </audio>
                    </div>
                  )}

                  <div className="mt-4 text-center">
                    <button onClick={playAudio} className="btn btn-info btn-lg me-3">
                      <i className="fas fa-play me-2"></i>
                      Play Audio
                    </button>
                    <button onClick={handleDownloadReport} className="btn btn-primary btn-lg me-3">
                      <i className="fas fa-download me-2"></i>
                      Download Report (TXT)
                    </button>
                    {recordedAudioURL && (
                      <button onClick={handleDownloadRecording} className="btn btn-success btn-lg me-3">
                        <i className="fas fa-file-audio me-2"></i>
                        Download Recorded Audio
                      </button>
                    )}
                    <button onClick={handleReset} className="btn btn-outline-secondary btn-lg">
                      <i className="fas fa-redo me-2"></i>
                      Analyze Another File
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-dark text-white py-4 mt-5">
        <div className="container">
          <div className="row">
            <div className="col-md-6">
              <h5><i className="fas fa-shield-alt me-2"></i>DeepFakeGuard</h5>
              <p className="mb-0">DeepFake Speech Detection System</p>
            </div>
            <div className="col-md-6 text-md-end">
              <p className="mb-0">
                <small>
                  <i className="fas fa-brain me-1"></i>
                  Powered by TensorFlow & Advanced CNN
                  <br />
                  Accuracy: 90% • F1-Score: 89.8%
                </small>
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
