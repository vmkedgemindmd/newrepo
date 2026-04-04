# 🍅 TomatoAI — Leaf Disease Detection

An AI-powered web application that detects **10 types of tomato leaf diseases** using a fine-tuned Vision Transformer (ViT) deep learning model with **99.67% accuracy**.

Upload a photo of a tomato leaf → Get an instant diagnosis with treatment recommendations.

---

## 🌟 Features

- **Real AI Model** — ViT (Vision Transformer) fine-tuned on PlantVillage dataset
- **99.67% Accuracy** — Trained on thousands of expert-labeled leaf images
- **10 Disease Classes** — Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
- **Smart Validation** — Rejects non-leaf images (human photos, objects, blanks)
- **Treatment Info** — Severity, description, and treatment for each disease
- **Dual Deployment** — Works on Vercel (HF API) or Render (local model)
- **Modern UI** — Dark theme, glassmorphism, drag-and-drop upload

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 19 + Vite 5 |
| Backend | FastAPI (Python) |
| ML Model | ViT (`google/vit-base-patch16-224`) fine-tuned on PlantVillage |
| ML Framework | PyTorch + Hugging Face Transformers |
| Image Processing | OpenCV, Pillow |

---

## 📦 Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Node.js 18+** — [nodejs.org](https://nodejs.org/)
- **Git** — [git-scm.com](https://git-scm.com/)

Verify in PowerShell:
```powershell
python --version
node --version
npm --version
```

---

## 🚀 Local Setup (Step by Step)

### Step 1: Clone the Repository

```powershell
git clone https://github.com/vmkedgemindmd/newrepo.git
cd newrepo
```

### Step 2: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

Then install PyTorch and Transformers for local model:

```powershell
pip install torch transformers
```

### Step 3: Download the ML Model (Auto)

The model (~343 MB) downloads **automatically** on first start. To pre-download:

```powershell
python -c "from transformers import pipeline; pipeline('image-classification', model='wellCh4n/tomato-leaf-disease-classification-vit', device='cpu'); print('Done!')"
```

> The model caches at `C:\Users\<YOU>\.cache\huggingface\hub\` — only downloads once.

### Step 4: Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### Step 5: Start the Backend (Terminal 1)

```powershell
python -m uvicorn api.index:app --reload --port 8000
```

Wait for:
```
Model loaded successfully!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 6: Start the Frontend (Terminal 2)

```powershell
cd frontend
npx vite --port 5173
```

### Step 7: Open the App

```
http://localhost:5173/
```

🎉 **Upload a tomato leaf image and get your diagnosis!**

---

## 🌐 Deployment Options

The backend supports **two modes** controlled by the `USE_HF_API` environment variable:

| Mode | `USE_HF_API` | How It Works | Best For |
|------|-------------|-------------|----------|
| **Local Model** | `false` (default) | Loads ViT model into memory | Local dev, Render |
| **HF API** | `true` | Calls Hugging Face Inference API | Vercel |

---

### Option A: Deploy on Vercel (HF API Mode)

Vercel can't load the large PyTorch model, so it calls the Hugging Face API instead.

#### Step 1: Install Vercel CLI

```powershell
npm install -g vercel
```

#### Step 2: Login to Vercel

```powershell
vercel login
```

#### Step 3: Get a Hugging Face API Token (Free)

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Name it `tomatoai`, select **"Read"** access
4. Copy the token (starts with `hf_...`)

#### Step 4: Set Environment Variables on Vercel

```powershell
cd d:\leaf_detection\newrepo
vercel env add USE_HF_API
# When prompted, enter: true
# Select: Production, Preview, Development

vercel env add HF_API_TOKEN
# When prompted, paste your hf_... token
# Select: Production, Preview, Development
```

#### Step 5: Deploy

```powershell
vercel --prod
```

#### How It Works on Vercel

```
User → Vercel Frontend (React)
         ↓
       Vercel Serverless Function (FastAPI, lightweight)
         ↓
       Hugging Face Inference API (runs the real ViT model)
         ↓
       Returns diagnosis back to user
```

> **Note:** The HF free API has rate limits. For production use, consider upgrading to HF Pro ($9/month) or using Render instead.

---

### Option B: Deploy on Render (Local Model, Recommended)

Render runs the full model locally — no API limits, full control.

#### Step 1: Create a Render Account

Go to [render.com](https://render.com/) and sign up (free).

#### Step 2: Connect Your GitHub Repo

1. In Render dashboard, click **"New" → "Web Service"**
2. Connect your GitHub account
3. Select the `newrepo` repository

#### Step 3: Configure the Service

| Setting | Value |
|---------|-------|
| **Name** | `tomatoai-backend` |
| **Region** | Pick closest to you |
| **Branch** | `main` |
| **Root Directory** | _(leave empty)_ |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements-render.txt` |
| **Start Command** | `uvicorn api.index:app --host 0.0.0.0 --port $PORT` |

#### Step 4: Add Environment Variable

In **Environment** tab:
| Key | Value |
|-----|-------|
| `USE_HF_API` | `false` |

#### Step 5: Deploy

Click **"Create Web Service"**. Render will:
1. Install dependencies (~5 min)
2. Download the ViT model (~2 min)
3. Start the server

You'll get a URL like: `https://tomatoai-backend.onrender.com`

#### Step 6: Deploy Frontend on Vercel

Now point your frontend to the Render backend:

```powershell
cd d:\leaf_detection\newrepo

# Set the API URL to your Render backend
vercel env add VITE_API_URL
# Enter: https://tomatoai-backend.onrender.com
# Select: Production, Preview, Development

# Deploy frontend
vercel --prod
```

#### Step 7: Update Frontend Vite Config for Production

The frontend already reads `VITE_API_URL` from environment:
```javascript
// In App.jsx — this line handles it:
const API_BASE = import.meta.env.VITE_API_URL || '';
```

#### How It Works on Render

```
User → Vercel Frontend (React)
         ↓
       Render Backend (FastAPI + ViT model loaded in memory)
         ↓
       Returns diagnosis back to user
```

> **Note:** Render free tier sleeps after 15 min of inactivity. First request after sleep takes ~30s to load the model.

---

## 📁 Project Structure

```
newrepo/
├── api/
│   └── index.py                # FastAPI backend (dual-mode)
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── index.css           # Dark theme styles
│   │   ├── main.jsx            # Entry point
│   │   └── components/
│   │       └── ProbabilityChart.jsx
│   ├── index.html
│   ├── vite.config.js          # Vite + API proxy
│   └── package.json
├── requirements.txt            # Vercel deps (lightweight)
├── requirements-render.txt     # Render deps (full, with PyTorch)
├── vercel.json                 # Vercel deployment config
├── render.yaml                 # Render deployment config
├── package.json                # Root scripts
└── README.md
```

---

## 🧠 ML Model Details

| Property | Value |
|----------|-------|
| Architecture | Vision Transformer (ViT) |
| Base Model | `google/vit-base-patch16-224-in21k` |
| Training Data | PlantVillage Tomato Leaf Disease Dataset |
| Accuracy | **99.67%** |
| Loss | 0.0170 |
| Input Size | 224 × 224 px |
| Classes | 10 |
| Source | [HuggingFace](https://huggingface.co/wellCh4n/tomato-leaf-disease-classification-vit) |

### Disease Classes

| # | Disease | Severity |
|---|---------|----------|
| 1 | Bacterial Spot | 🔴 High |
| 2 | Early Blight | 🟡 Medium |
| 3 | Late Blight | 🔴 Critical |
| 4 | Leaf Mold | 🟡 Medium |
| 5 | Septoria Leaf Spot | 🟡 Medium |
| 6 | Spider Mites | 🟡 Medium |
| 7 | Target Spot | 🟡 Medium |
| 8 | Yellow Leaf Curl Virus | 🔴 Critical |
| 9 | Mosaic Virus | 🔴 High |
| 10 | Healthy | 🟢 None |

---

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + model info |
| `POST` | `/api/predict` | Upload image → get diagnosis |

### PowerShell API Test

```powershell
# Health check
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/health | ConvertTo-Json

# Predict
$form = @{ file = Get-Item "C:\path\to\leaf.jpg" }
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/predict -Method Post -Form $form | ConvertTo-Json -Depth 5
```

---

## 🤝 Credits

- **Model**: [wellCh4n/tomato-leaf-disease-classification-vit](https://huggingface.co/wellCh4n/tomato-leaf-disease-classification-vit)
- **Dataset**: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
- **Base Model**: [Google ViT](https://huggingface.co/google/vit-base-patch16-224-in21k)

## 📄 License

Educational use. ML model is Apache 2.0 licensed.
