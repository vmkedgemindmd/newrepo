import { useState, useCallback, useRef } from 'react';
import ProbabilityChart from './components/ProbabilityChart';
import './index.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

const DISEASE_ICONS = {
  'Bacterial Spot': '🦠',
  'Early Blight': '🍂',
  'Late Blight': '⚠️',
  'Leaf Mold': '🍄',
  'Septoria Leaf Spot': '🔴',
  'Spider Mites': '🕷️',
  'Target Spot': '🎯',
  'Yellow Leaf Curl Virus': '🌿',
  'Mosaic Virus': '🧬',
  'Healthy': '✅',
};

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('online');

  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError({ type: 'format', message: 'Please upload a valid image file (JPG, PNG, JPEG).' });
      return;
    }
    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);

  const handleFileInput = (e) => handleFile(e.target.files[0]);

  const clearImage = () => {
    setImageFile(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!imageFile) return;
    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Server responded with status ${res.status}`);

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError({
        type: 'network',
        message: err.message.includes('fetch') 
          ? 'Could not connect to the analysis server. The API might be offline or blocked by a firewall.'
          : `Analysis Failed: ${err.message}`,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const isHealthy = result?.diagnosis === 'Healthy';
  const diagClass = isHealthy ? 'healthy' : 'diseased';

  return (
    <div className="app-wrapper">
      {/* Background Orbs */}
      <div className="bg-orbs">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="header-inner">
            <div className="logo">
              <div className="logo-icon">🍅</div>
              <div className="logo-text">
                TomatoAI
                <span>Leaf Disease Detection</span>
              </div>
            </div>
            <div className="status-badge">
              <div className="status-dot" />
              AI Engine Active
            </div>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero-badge">
            ✨ Powered by Xception Deep Learning
          </div>
          <h1>
            Detect Tomato Leaf
            <br />
            <span className="highlight">Diseases Instantly</span>
          </h1>
          <p>
            Upload a photo of your tomato plant leaf and get an instant AI-powered diagnosis
            across 10 disease categories with treatment recommendations.
          </p>
        </div>
      </section>

      {/* Stats Row */}
      <div className="container">
        <div className="stats-row">
          {[
            { num: '10', label: 'Disease Classes' },
            { num: '99%', label: 'Accuracy Rate' },
            { num: '<1s', label: 'Detection Time' },
          ].map(({ num, label }) => (
            <div key={label} className="glass-card stat-card">
              <div className="stat-number">{num}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <main className="container">
        <div className="main-grid">
          {/* Upload Card */}
          <div className="glass-card">
            <div className="card-header">
              <div className="card-icon blue">📤</div>
              <div>
                <div className="card-title">Upload Leaf Image</div>
                <div className="card-subtitle">JPG, PNG, or JPEG supported</div>
              </div>
            </div>
            <div className="upload-area">
              {/* Drop Zone */}
              <div
                className={`drop-zone ${isDragging ? 'drag-over' : ''} ${imageUrl ? 'has-image' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={!imageUrl ? () => fileInputRef.current?.click() : undefined}
              >
                {imageUrl ? (
                  <img src={imageUrl} alt="Leaf preview" className="preview-img" />
                ) : (
                  <>
                    <div className="upload-icon">🌿</div>
                    <h3>Drop your leaf image here</h3>
                    <p>
                      or <span onClick={() => fileInputRef.current?.click()}>browse files</span> from your device
                    </p>
                  </>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                style={{ display: 'none' }}
                id="leaf-file-input"
              />

              {imageFile && (
                <div className="file-info">
                  <span className="file-name">{imageFile.name}</span>
                  <button className="file-remove-btn" onClick={clearImage}>
                    ✕ Remove
                  </button>
                </div>
              )}

              <button
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={!imageFile || isAnalyzing}
                id="analyze-btn"
              >
                {isAnalyzing ? (
                  <>
                    <div className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    🔬 Analyze Leaf
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Card */}
          <div className="glass-card">
            <div className="card-header">
              <div className="card-icon red">🧪</div>
              <div>
                <div className="card-title">Diagnosis Result</div>
                <div className="card-subtitle">AI-powered disease classification</div>
              </div>
            </div>
            <div className="result-panel">
              {/* No result yet */}
              {!result && !error && !isAnalyzing && (
                <div className="empty-state">
                  <div className="empty-icon">🔎</div>
                  <h3>Awaiting Analysis</h3>
                  <p>Upload an image and click Analyze Leaf to get your diagnosis.</p>
                </div>
              )}

              {/* Loading */}
              {isAnalyzing && (
                <div className="empty-state">
                  <div className="empty-icon" style={{ animation: 'orbFloat 2s ease-in-out infinite' }}>🌿</div>
                  <h3>Analyzing your leaf...</h3>
                  <p>Our AI is inspecting for signs of disease.</p>
                </div>
              )}

              {/* Error: not a leaf */}
              {error && error.type !== 'format' && !result && (
                <div className="error-state">
                  <div className="error-icon">⚠️</div>
                  <h3>Analysis Failed</h3>
                  <p>{error.message}</p>
                </div>
              )}

              {/* Server not found error */}
              {error && error.type === 'format' && (
                <div className="error-state">
                  <div className="error-icon">🚫</div>
                  <h3>Invalid File</h3>
                  <p>{error.message}</p>
                </div>
              )}

              {/* Not a leaf detection */}
              {result && !result.success && (
                <div className="error-state">
                  <div className="error-icon">🚫</div>
                  <h3>Not a Tomato Leaf</h3>
                  <p style={{ marginBottom: '12px' }}>{result.message}</p>
                  <div style={{
                    marginTop: '12px',
                    padding: '10px 14px',
                    background: 'rgba(255,255,255,0.04)',
                    borderRadius: '10px',
                    fontSize: '0.78rem',
                    color: 'var(--text-muted)',
                    textAlign: 'left',
                    lineHeight: '1.7'
                  }}>
                    <strong style={{ color: 'var(--text-secondary)' }}>✅ Valid inputs:</strong><br />
                    • Close-up photo of a tomato leaf<br />
                    • Clear green, yellow, or diseased leaf<br />
                    • Natural lighting, sharp focus<br /><br />
                    <strong style={{ color: 'var(--text-secondary)' }}>❌ Invalid inputs:</strong><br />
                    • Advertisements, posters, or banners<br />
                    • Photos of objects, people, or buildings<br />
                    • Blank or near-white backgrounds
                  </div>
                </div>
              )}

              {/* Successful result */}
              {result && result.success && (
                <div className="diagnosis-area">
                  <div className="diagnosis-header">
                    <div className={`diagnosis-icon ${diagClass}`}>
                      {DISEASE_ICONS[result.diagnosis] || '🌿'}
                    </div>
                    <div>
                      <div className={`diagnosis-name ${diagClass}`}>
                        {result.diagnosis}
                      </div>
                      <div className="confidence-row">
                        <span className={`confidence-pill ${diagClass}`}>
                          {result.confidence?.toFixed(1)}% confidence
                        </span>
                        {result.severity && (
                          <span className={`severity-badge severity-${result.severity}`}>
                            {result.severity}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Confidence bar */}
                  <div className="confidence-bar-wrap">
                    <div className="confidence-bar-label">
                      <span>Confidence Level</span>
                      <span>{result.confidence?.toFixed(1)}%</span>
                    </div>
                    <div className="confidence-bar-track">
                      <div
                        className={`confidence-bar-fill ${diagClass}`}
                        style={{ width: `${result.confidence}%` }}
                      />
                    </div>
                  </div>

                  {/* Info grid */}
                  <div className="info-grid">
                    <div className="info-item">
                      <div className="info-item-label">Description</div>
                      <div className="info-item-value">{result.description}</div>
                    </div>
                    <div className="info-item">
                      <div className="info-item-label">Treatment</div>
                      <div className="info-item-value">{result.treatment}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Probability Chart — Full Width */}
          {result && result.success && (
            <div className="glass-card chart-section">
              <div className="card-header">
                <div className="card-icon orange">📊</div>
                <div>
                  <div className="card-title">Prediction Probability Distribution</div>
                  <div className="card-subtitle">Confidence across all 10 disease classes</div>
                </div>
              </div>
              <div className="chart-body">
                <ProbabilityChart
                  probabilities={result.probabilities}
                  topClass={result.diagnosis}
                />
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>🍅 TomatoAI — AI-Based Tomato Leaf Disease Detection</p>
          <p style={{ marginTop: '6px' }}>
            Built with FastAPI + React · Xception Deep Learning Model
          </p>
        </div>
      </footer>
    </div>
  );
}
