import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { 
  Upload, Music, Download, RefreshCw, 
  Loader, CheckCircle, XCircle, Image as ImageIcon, ArrowLeft 
} from 'lucide-react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [midiUrl, setMidiUrl] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [requestId, setRequestId] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [duration, setDuration] = useState(15);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setError(null);
      setSuccess(false);
      setMidiUrl(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    },
    maxFiles: 1
  });

  // Generate music from image
  const generateMusic = async () => {
    if (!image) {
      setError('Please upload an image first');
      return;
    }

    setIsGenerating(true);
    setError(null);
    setSuccess(false);
    setMidiUrl(null);

    const formData = new FormData();
    formData.append('file', image);
    formData.append('duration', duration);

    try {
      const response = await axios.post('/api/generate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setMidiUrl(response.data.midi_url);
        setAudioUrl(response.data.audio_url || null);
        setRequestId(response.data.request_id);
        setSuccess(true);
      } else {
        setError('Failed to generate music');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while generating music');
      console.error('Error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  // Download audio file (MP3/WAV) or MIDI as fallback
  const downloadAudio = () => {
    const url = audioUrl || midiUrl;
    const extension = audioUrl ? '.wav' : '.mid';
    if (url) {
      const link = document.createElement('a');
      link.href = `http://localhost:8000${url}`;
      link.download = `lofi_${requestId}${extension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Download MIDI file specifically
  const downloadMidi = () => {
    if (midiUrl) {
      const link = document.createElement('a');
      link.href = `http://localhost:8000${midiUrl}`;
      link.download = `lofi_${requestId}.mid`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Go back to upload
  const goBackToUpload = () => {
    setImage(null);
    setImagePreview(null);
    setError(null);
  };

  // Reset to start over
  const reset = () => {
    setImage(null);
    setImagePreview(null);
    setMidiUrl(null);
    setAudioUrl(null);
    setRequestId(null);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Music className="logo-icon" />
            <h1>Lofi Generator</h1>
          </div>
          <p className="tagline">Transform images into lofi beats with AI</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          
          {/* Upload Section */}
          {!imagePreview && (
            <div className="upload-section">
              <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <div className="dropzone-content">
                  <Upload className="dropzone-icon" size={64} />
                  <h2>Upload an Image</h2>
                  <p>
                    {isDragActive
                      ? 'Drop your image here...'
                      : 'Drag & drop an image, or click to browse'}
                  </p>
                  <div className="supported-formats">
                    <span>Supports: JPG, PNG, GIF, WebP</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Preview and Generate Section */}
          {imagePreview && !success && (
            <div className="preview-section">
              <button className="back-button" onClick={goBackToUpload}>
                <ArrowLeft size={20} />
                Back to Upload
              </button>

              <div className="image-preview-container">
                <img src={imagePreview} alt="Preview" className="image-preview" />
                <button className="remove-image" onClick={goBackToUpload} title="Remove image">
                  <XCircle size={24} />
                </button>
              </div>

              <div className="controls">
                <div className="duration-control">
                  <label htmlFor="duration">Duration (seconds):</label>
                  <input
                    id="duration"
                    type="range"
                    min="10"
                    max="60"
                    value={duration}
                    onChange={(e) => setDuration(parseInt(e.target.value))}
                    disabled={isGenerating}
                  />
                  <span className="duration-value">{duration}s</span>
                </div>

                <button
                  className="generate-button"
                  onClick={generateMusic}
                  disabled={isGenerating}
                >
                  {isGenerating ? (
                    <>
                      <Loader className="spinner" size={20} />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Music size={20} />
                      Generate Lofi Beat
                    </>
                  )}
                </button>
              </div>

              {error && (
                <div className="alert alert-error">
                  <XCircle size={20} />
                  <span>{error}</span>
                </div>
              )}
            </div>
          )}

          {/* Success Section */}
          {success && midiUrl && (
            <div className="success-section">
              <button className="back-button" onClick={reset}>
                <ArrowLeft size={20} />
                Start Over
              </button>

              <div className="success-header">
                <CheckCircle className="success-icon" size={64} />
                <h2>Your Lofi Beat is Ready!</h2>
                <p>Download your unique AI-generated music</p>
              </div>

              <div className="result-preview">
                <img src={imagePreview} alt="Source" className="result-image" />
              </div>

              <div className="action-buttons">
                <button className="download-button" onClick={downloadAudio}>
                  <Download size={20} />
                  Download {audioUrl ? 'Audio (WAV)' : 'MIDI'}
                </button>
                {audioUrl && (
                  <button className="download-button" onClick={downloadMidi} style={{background: 'var(--bg-tertiary)', border: '1px solid var(--border)'}}>
                    <Download size={20} />
                    Download MIDI
                  </button>
                )}
                <button className="reset-button" onClick={reset}>
                  <RefreshCw size={20} />
                  Create Another
                </button>
              </div>

              <div className="midi-info">
                <p>
                  <strong>Note:</strong> {audioUrl 
                    ? 'Downloaded as WAV audio file. You can also download the MIDI file separately to edit in any DAW.' 
                    : 'This is a MIDI file. You can open it in any DAW (Digital Audio Workstation) like GarageBand, FL Studio, Ableton, or use online MIDI players to listen to it.'}
                </p>
              </div>
            </div>
          )}

          {/* How It Works */}
          {!imagePreview && (
            <div className="how-it-works">
              <h2>How It Works</h2>
              <div className="steps">
                <div className="step">
                  <div className="step-number">1</div>
                  <ImageIcon size={32} />
                  <h3>Upload Image</h3>
                  <p>Choose any image that captures a vibe or mood</p>
                </div>
                <div className="step">
                  <div className="step-number">2</div>
                  <Music size={32} />
                  <h3>AI Generation</h3>
                  <p>Our AI analyzes colors, contrast, and patterns</p>
                </div>
                <div className="step">
                  <div className="step-number">3</div>
                  <Download size={32} />
                  <h3>Download</h3>
                  <p>Get your unique lofi beat as a MIDI/WAV file</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Built with React, FastAPI, and Neural Networks</p>
      </footer>
    </div>
  );
}

export default App;
