# Fish Origin Classification - Deployment Guide

This application has been refactored into a **standalone Streamlit app** with integrated model inference (no separate Flask API needed).

## Local Development

### Prerequisites
- Python 3.8+
- Model checkpoint file: `best.ckpt` in project root

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open browser at: `http://localhost:8501`

---

## Deployment to Streamlit Cloud

### Step 1: Prepare Repository

1. Push code to GitHub repository
2. Ensure these files are present:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `best.ckpt` (model checkpoint)
   - `src/` folder (model and utility modules)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.9 or 3.10
6. Click "Deploy"

### Step 3: Monitor Deployment

- Streamlit Cloud will automatically install dependencies from `requirements.txt`
- Model will be loaded on first access (cached for subsequent requests)
- Check logs for any errors

---

## Deployment to Other Platforms

### Option 1: Railway

1. Install Railway CLI or use web dashboard
2. Create new project from GitHub repo
3. Railway will auto-detect Python app
4. Set start command:
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
5. Deploy

### Option 2: Heroku

1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

2. Create `runtime.txt`:
   ```
   python-3.10.12
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

Build and run with Docker:

```bash
docker build -t fish-classifier .
docker run -p 8501:8501 fish-classifier
```

See `Dockerfile` in repository.

---

## Configuration

### Memory Requirements

- Model size: ~50MB (checkpoint file)
- RAM needed: ~2GB minimum (for model + inference)
- Streamlit Cloud free tier: 1GB RAM (may need optimization)

### Environment Variables

None required for basic deployment. Optional:

- `STREAMLIT_SERVER_PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

---

## Troubleshooting

### Issue: Out of Memory on Streamlit Cloud

**Solution**: Streamlit Cloud free tier has 1GB RAM limit. Options:
1. Quantize model to reduce size
2. Use model pruning
3. Upgrade to paid tier
4. Deploy to alternative platform (Railway, Render)

### Issue: Model Loading Slow

**Solution**: Model is cached after first load using `@st.cache_resource`. First request will be slow, subsequent requests are fast.

### Issue: Large Checkpoint File

**Solution**: Use Git LFS for files >100MB:
```bash
git lfs install
git lfs track "*.ckpt"
git add .gitattributes
git add best.ckpt
git commit -m "Add model checkpoint with LFS"
```

---

## Performance Notes

- **First run**: 5-10 seconds (model loading)
- **Subsequent runs**: <1 second per image (cached model)
- **Batch inference**: Processes images sequentially
- **Grad-CAM generation**: Adds ~0.5s per image

---

## Architecture Changes

### Before (Flask + Streamlit)
```
User → Streamlit UI → HTTP Request → Flask API → Model → Response
```

### After (Streamlit Only)
```
User → Streamlit UI → Direct Function Call → Model → Response
```

**Benefits**:
- Simpler deployment
- No HTTP overhead
- Easier to scale
- Free hosting on Streamlit Cloud

**Trade-offs**:
- No separate API endpoint (if other apps need it, would need to deploy separately)
- Streamlit session-based (not stateless like API)

---

## File Structure

```
salmon test web/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── best.ckpt              # Model checkpoint
├── src/
│   ├── config.py          # Configuration constants
│   ├── utils.py           # Utility functions
│   ├── image_preprocessing.py
│   └── model/
│       └── simple_multimodal_cnn.py
├── README_DEPLOYMENT.md   # This file
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

---

## Support

For issues or questions:
1. Check Streamlit Cloud logs
2. Verify all dependencies are installed
3. Ensure `best.ckpt` is accessible
4. Check model architecture matches checkpoint

---

*Last updated: 2025-01-20*
