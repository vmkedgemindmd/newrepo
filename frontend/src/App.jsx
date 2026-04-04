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

const STATS = [
  { value: '10', label: 'Disease Classes' },
  { value: 'CV', label: 'Detection Engine' },
  { value: '<1s', label: 'Analysis Time' },
];

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // ── Handle file selection ──────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setError({ type: 'format', message: 'Please upload a valid image file (JPG, PNG, JPEG, WEBP).' });
      setImageFile(null);
      setImageUrl(null);
      setResult(null);
      return;
    }

    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  // ── Clear everything ───────────────────────────────────────────
  const clearImage = useCallback(() => {
    setImageFile(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  // ── Drag & Drop handlers ──────────────────────────────────────
  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  };

  // ── Analyze leaf ───────────────────────────────────────────────
  const analyzeLeaf = async () => {
    if (!imageFile) return;

    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server responded with status ${res.status}`);
      }

      const data = await res.json();

      if (data.success) {
        setResult(data);
        setError(null);
      } else {
        setResult(null);
        setError({ type: 'not_leaf', message: data.message });
      }
    } catch (err) {
      const msg = err.message || 'Unknown error';
      if (msg.includes('fetch') || msg.includes('Failed') || msg.includes('NetworkError')) {
        setError({
          type: 'network',
          message: 'Could not connect to the analysis server. Make sure the API is running (uvicorn api.index:app --reload).',
        });
      } else {
        setError({ type: 'server', message: `Analysis failed: ${msg}` });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const diagClass = result
    ? result.diagnosis === 'Healthy' ? 'healthy' : 'disease'
    : '';

  return (
    <div className="app">
      {/* Background orbs */}
      <div className="bg-orbs">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      {/* Navbar */}
      <nav className="navbar">
        <div className="container navbar-inner">
          <div className="nav-brand">
            <span className="nav-logo">🍅</span>
            <span className="nav-title">TomatoAI</span>
          </div>
          <div className="nav-links">
            <span className="nav-link active">Detect</span>
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="main">
        <div className="container">

          {/* Hero */}
          <section className="hero">
            <div className="hero-badge">
              <span className="hero-badge-dot" />
              AI-Powered Disease Detection
            </div>
            <h1 className="hero-title">
              Detect Tomato Leaf<br />
              Diseases <span className="gradient-text">Instantly</span>
            </h1>
            <p className="hero-subtitle">
              Upload a photo of your tomato plant leaf and get an instant AI-powered diagnosis
              with treatment recommendations.
            </p>

            {/* Stats row */}
            <div className="stats-row">
              {STATS.map(({ value, label }) => (
                <div key={label} className="glass-card stat-card">
                  <div className="stat-value">{value}</div>
                  <div className="stat-label">{label}</div>
                </div>
              ))}
            </div>
          </section>

          {/* Two-column layout */}
          <div className="grid-2col">

            {/* Upload Card */}
            <div className="glass-card">
              <div className="card-header">
                <div className="card-icon blue">📤</div>
                <div>
                  <div className="card-title">Upload Leaf Image</div>
                  <div className="card-subtitle">JPG, PNG, or JPEG supported</div>
                </div>
              </div>

              {/* Drop zone */}
              <div
                className={`drop-area ${isDragging ? 'dragging' : ''} ${imageUrl ? 'has-image' : ''}`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => !imageUrl && fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/jpg,image/webp"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFile(e.target.files[0])}
                />

                {imageUrl ? (
                  <div className="preview-wrap">
                    <img src={imageUrl} alt="Uploaded leaf" className="preview-img" />
                    <div className="preview-info">
                      <span className="preview-name">{imageFile.name}</span>
                      <button className="btn-clear" onClick={(e) => { e.stopPropagation(); clearImage(); }}>✕ Remove</button>
                    </div>
                  </div>
                ) : (
                  <div className="drop-placeholder">
                    <div className="drop-icon">🌿</div>
                    <p>Drag & drop a leaf image here</p>
                    <p className="drop-hint">
                      or <span className="drop-link" onClick={() => fileInputRef.current?.click()}>browse files</span> from your device
                    </p>
                  </div>
                )}
              </div>

              {/* Format error */}
              {error && error.type === 'format' && (
                <div className="inline-error">⚠️ {error.message}</div>
              )}

              {/* Analyze Button */}
              <button
                className="btn-analyze"
                disabled={!imageFile || isAnalyzing}
                onClick={analyzeLeaf}
              >
                {isAnalyzing ? (
                  <>
                    <span className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>🔬 Analyze Leaf</>
                )}
              </button>

              {/* Tips */}
              <div className="tips-box">
                <strong style={{ color: 'var(--text-secondary)' }}>✅ Valid inputs:</strong><br />
                • Close-up photo of an actual tomato leaf<br />
                • Natural lighting, sharp focus<br /><br />
                <strong style={{ color: 'var(--text-secondary)' }}>❌ Invalid inputs:</strong><br />
                • Advertisements, posters, or banners<br />
                • Photos of objects, people, or buildings<br />
                • Blank or near-white backgrounds
              </div>
            </div>

            {/* Results Card */}
            <div className="glass-card">
              <div className="card-header">
                <div className="card-icon green">🧪</div>
                <div>
                  <div className="card-title">Analysis Results</div>
                  <div className="card-subtitle">Diagnosis & treatment recommendations</div>
                </div>
              </div>

              {/* Loading state */}
              {isAnalyzing && (
                <div className="loading-state">
                  <div className="loading-spinner-lg" />
                  <h3>Analyzing Leaf...</h3>
                  <p>Processing image features and running classification...</p>
                </div>
              )}

              {/* Empty state */}
              {!result && !error && !isAnalyzing && (
                <div className="empty-state">
                  <div className="empty-icon">🔎</div>
                  <h3>Awaiting Analysis</h3>
                  <p>Upload an image and click Analyze Leaf to get your diagnosis.</p>
                </div>
              )}

              {/* Not-a-leaf error */}
              {error && error.type === 'not_leaf' && !isAnalyzing && (
                <div className="error-state">
                  <div className="error-icon">🚫</div>
                  <h3>Not a Tomato Leaf</h3>
                  <p style={{ marginBottom: '12px' }}>{error.message}</p>
                  <p style={{
                    fontSize: '0.85rem',
                    color: 'var(--text-muted)',
                    textAlign: 'left',
                    lineHeight: 1.7,
                  }}>
                    <strong>Tips for a valid scan:</strong><br />
                    • Use a close-up of an actual tomato leaf<br />
                    • Avoid pictures of objects, ads, or blank images<br />
                    • Ensure good lighting and focus
                  </p>
                </div>
              )}

              {/* Network / server error */}
              {error && (error.type === 'network' || error.type === 'server') && !isAnalyzing && (
                <div className="error-state">
                  <div className="error-icon">⚠️</div>
                  <h3>Connection Error</h3>
                  <p>{error.message}</p>
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
            Built with FastAPI + React · Computer Vision Analysis Engine
          </p>
        </div>
      </footer>
    </div>
  );
}
