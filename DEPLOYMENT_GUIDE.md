# 🚀 Deployment Guide

## Problem: Build Fails Due to Disk Space

The original `requirements.txt` includes PyTorch with CUDA support, which is **~3GB+** in size. This causes deployment platforms to run out of disk space.

## Solution: Use Production Requirements

### Option 1: Use CPU-Only PyTorch (Recommended)

Use `requirements-prod.txt` with CPU-only PyTorch (~200MB instead of ~3GB):

**Method 1: Use install script (easiest)**
```bash
# Linux/Mac
bash install-prod.sh

# Windows
install-prod.bat
```

**Method 2: Manual installation**
```bash
# First install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements-prod.txt
```

**Benefits:**
- ✅ Much smaller (~200MB vs ~3GB)
- ✅ Works on any platform (no GPU needed)
- ✅ Faster installation
- ✅ Sufficient for inference (model predictions)

**Limitations:**
- ❌ No GPU acceleration (but fine for web app inference)
- ❌ Slower training (but training should be done locally anyway)

### Option 2: Make ML Dependencies Optional

The app already handles missing ML libraries gracefully:
- CNN/YOLO models load lazily in background
- If models fail to load, app still runs (uses database matching)
- Face recognition detection is optional

You can remove PyTorch entirely if you only use database matching:

```txt
Flask==3.0.3
Werkzeug==3.0.4
Pillow>=8.0.0
numpy>=1.21.0
supabase>=2.0.0
python-dotenv>=1.0.0
scipy>=1.10.0
```

## Platform-Specific Deployment

### Netlify

1. **Create `netlify.toml`** (already created):
```toml
[build]
  command = "pip install -r requirements-prod.txt"
```

2. **Set Environment Variables** in Netlify Dashboard:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `FLASK_ENV=production`
   - `FLASK_SECRET_KEY` (generate a random secret)

3. **Note**: Netlify is primarily for static sites. For Flask apps, consider:
   - **Render.com** (recommended for Flask)
   - **Railway.app**
   - **Fly.io**
   - **Heroku** (paid)

### Render.com (Recommended for Flask)

1. **Create `render.yaml`**:
```yaml
services:
  - type: web
    name: pechay-detection
    env: python
    buildCommand: pip install -r requirements-prod.txt
    startCommand: gunicorn app:app
    envVars:
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_ANON_KEY
        sync: false
      - key: SUPABASE_SERVICE_ROLE_KEY
        sync: false
      - key: FLASK_SECRET_KEY
        generateValue: true
```

2. **Install Gunicorn** (add to `requirements-prod.txt`):
```txt
gunicorn>=21.0.0
```

3. **Update start command**:
```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

### Railway.app

1. **Create `Procfile`**:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

2. **Set build command**:
```bash
pip install -r requirements-prod.txt && pip install gunicorn
```

### Vercel (Serverless Functions)

For Vercel, you'll need to convert Flask routes to serverless functions. Consider using **Render** or **Railway** instead for easier Flask deployment.

## Quick Fix: Update requirements.txt

If you want to use the production requirements by default, rename:

```bash
mv requirements.txt requirements-dev.txt
mv requirements-prod.txt requirements.txt
```

## Environment Variables Needed

Make sure to set these in your deployment platform:

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Flask
FLASK_ENV=production
FLASK_SECRET_KEY=generate-a-random-secret-key-here

# Optional: Model paths (if using ML)
CNN_MODEL_PATHS=pechay_cnn_model_20251212_184656.pth
YOLO_WEIGHTS_PATH=weights/best.pt
```

## Testing Locally

Test the production requirements locally:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production requirements
pip install -r requirements-prod.txt

# Test the app
python app.py
```

## Recommended Deployment Platform

**For Flask apps, I recommend Render.com** because:
- ✅ Easy Flask deployment
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Environment variable management
- ✅ Good documentation

**Steps for Render:**
1. Connect your GitHub repo
2. Select "Web Service"
3. Use build command: `pip install -r requirements-prod.txt && pip install gunicorn`
4. Use start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add environment variables
6. Deploy!

## Troubleshooting

### "No space left on device"
- ✅ Use `requirements-prod.txt` (CPU-only PyTorch)
- ✅ Remove unnecessary dependencies (matplotlib, seaborn if not needed)

### "Module not found"
- ✅ Check if ML dependencies are optional (they are in this app)
- ✅ App will run without PyTorch/YOLO (uses database matching)

### "Port already in use"
- ✅ Use `$PORT` environment variable (platforms set this)
- ✅ Bind to `0.0.0.0` not `127.0.0.1`

## Next Steps

1. ✅ Use `requirements-prod.txt` for deployment
2. ✅ Choose a deployment platform (Render.com recommended)
3. ✅ Set environment variables
4. ✅ Deploy!

