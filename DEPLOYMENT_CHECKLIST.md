# Render.com Deployment Checklist ✅

## Pre-Deployment Verification

### 📋 Required Files Present
- [ ] `app_launcher.py` - Main coordinator application
- [ ] `app_chemberta.py` - ChemBERTa transformer application
- [ ] `app_rdkit.py` - RDKit descriptors application  
- [ ] `app_circular.py` - Circular fingerprints application
- [ ] `app_graph_combined.py` - Graph neural networks application
- [ ] `requirements.txt` - Python dependencies
- [ ] `Dockerfile` - Production container definition
- [ ] `render.yaml` - Render service configuration
- [ ] `.streamlit/config.toml` - Streamlit production settings

### 🧬 Model Files Present
- [ ] `best_model_aggregrate_circular.pkl`
- [ ] `bestPipeline_tpot_circularfingerprint_classification.pkl`
- [ ] `bestPipeline_tpot_rdkit_classification.pkl`
- [ ] `bestPipeline_tpot_rdkit_Regression.pkl`
- [ ] `train_data.pkl`
- [ ] `X_train_circular.pkl`
- [ ] `checkpoint-2000/` directory with all files
- [ ] `GraphConv_model_files/` directory
- [ ] `graphConv_reg_model_files 2/` directory

### ⚙️ Configuration Verified
- [ ] `render.yaml` uses correct plan (standard recommended)
- [ ] Dockerfile uses dynamic `$PORT` variable
- [ ] Health check path set to `/_stcore/health`
- [ ] Environment variables configured
- [ ] Disk storage configured (2GB recommended)
- [ ] No "AChE Inhibitor Prediction" headers in individual apps
- [ ] ChemBERTa app available in launcher
- [ ] All 4 apps load correctly from launcher
- [ ] Model files present and accessible
- [ ] Requirements.txt updated with correct versions

## GitHub Setup ✅

- [ ] Run `./setup_github.sh` to initialize repository
- [ ] Create new repository on GitHub
- [ ] Add remote origin: `git remote add origin https://github.com/USERNAME/REPO.git`
- [ ] Push to GitHub: `git push -u origin main`
- [ ] Verify all files uploaded correctly

## Render.com Deployment ✅

- [ ] Sign up/login to Render.com
- [ ] Connect GitHub account
- [ ] Create new Web Service
- [ ] Select your repository
- [ ] Render.com detects `render.yaml` automatically
- [ ] Review configuration and deploy
- [ ] Wait for build to complete (10-15 minutes)

## Post-Deployment Testing ✅

- [ ] App loads at Render URL
- [ ] Launcher shows all 4 app cards
- [ ] ChemBERTa app accessible
- [ ] RDKit app works (molecular prediction)
- [ ] Circular fingerprints app functions
- [ ] Graph neural networks app loads
- [ ] No broken imports or missing files
- [ ] All prediction functionalities work

## Production Considerations ✅

- [ ] Monitor app performance
- [ ] Check memory usage (upgrade plan if needed)
- [ ] Set up custom domain (optional)
- [ ] Enable auto-deploy for future updates
- [ ] Monitor Render.com logs for errors

## Files Included ✅

- [ ] `app_launcher.py` - Main launcher app
- [ ] `app_chemberta.py` - ChemBERTa transformer
- [ ] `app_rdkit.py` - RDKit molecular prediction
- [ ] `app_circular.py` - Circular fingerprints
- [ ] `app_graph_combined.py` - Graph neural networks
- [ ] `Dockerfile.launcher` - Docker configuration
- [ ] `requirements.txt` - Python dependencies
- [ ] `render.yaml` - Render.com configuration
- [ ] `.dockerignore` - Docker ignore rules
- [ ] `DEPLOYMENT.md` - Deployment guide
- [ ] `start_local.sh` - Local testing script
- [ ] `setup_github.sh` - GitHub setup script
- [ ] All model files (`.pkl`, checkpoints, etc.)

## Quick Commands

```bash
# Local testing
./start_local.sh

# GitHub setup
./setup_github.sh

# Manual git commands
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
```

## Expected URLs

- **Local**: http://localhost:8501
- **Render**: https://your-service-name.onrender.com

## Support

- Render.com docs: https://render.com/docs
- Streamlit docs: https://docs.streamlit.io
- Issues: Check build logs in Render dashboard
