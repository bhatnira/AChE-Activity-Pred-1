# Render.com Deployment Guide

## Overview
This guide will help you deploy the Molecular Prediction Suite to Render.com using Docker.

## Prerequisites
- Render.com account
- Git repository with your code
- All model files (.pkl) and checkpoint directories in your repository

## Quick Deploy Options

### Option 1: Deploy with render.yaml (Recommended)
1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Connect your repository to Render.com
3. Render will automatically detect the `render.yaml` file and deploy

### Option 2: Manual Web Service Creation
1. Go to Render Dashboard
2. Click "New" → "Web Service"
3. Connect your Git repository
4. Configure the service:

## Service Configuration

### Basic Settings
- **Name**: `molecular-prediction-suite`
- **Environment**: `Docker`
- **Plan**: `Standard` (recommended for ML workloads)
- **Region**: `Oregon` (or closest to your users)

### Build & Deploy
- **Dockerfile Path**: `./Dockerfile`
- **Docker Command**: `streamlit run app_launcher.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

### Environment Variables
```
PORT=10000
STREAMLIT_SERVER_PORT=$PORT
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_THEME_BASE=light
STREAMLIT_THEME_PRIMARY_COLOR=#007AFF
```

### Disk Storage
- **Name**: `model-data`
- **Mount Path**: `/app/data`
- **Size**: `2 GB`

## Files Structure for Deployment

Ensure these files are in your repository:

### Required Files
```
├── app_launcher.py              # Main coordinator
├── app_chemberta.py            # ChemBERTa application
├── app_rdkit.py                # RDKit application
├── app_circular.py             # Circular fingerprints
├── app_graph_combined.py       # Graph neural networks
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container definition
├── render.yaml                 # Render configuration
├── .streamlit/config.toml      # Streamlit settings
└── start-render.sh             # Startup script
```

### Model Files
```
├── *.pkl                       # Model pickle files
├── checkpoint-2000/            # ChemBERTa checkpoint
├── GraphConv_model_files/      # Graph model files
└── graphConv_reg_model_files 2/ # Graph regression models
```

## Deployment Steps

### Step 1: Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### Step 2: Deploy on Render
1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New"** → **"Web Service"**
3. **Connect Repository**: Choose your Git provider and repository
4. **Configure Service**:
   - Name: `molecular-prediction-suite`
   - Environment: `Docker`
   - Plan: `Standard`
   - Dockerfile Path: `./Dockerfile`

### Step 3: Set Environment Variables
In the Render dashboard, add these environment variables:
- `PORT`: `10000`
- `STREAMLIT_SERVER_HEADLESS`: `true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: `false`

### Step 4: Configure Health Check
- **Health Check Path**: `/_stcore/health`

### Step 5: Deploy
Click "Create Web Service" and wait for deployment to complete.

## Monitoring & Troubleshooting

### Logs
- View real-time logs in Render dashboard
- Check for any missing dependencies or model files

### Health Check
- Render automatically monitors `/_stcore/health`
- Service will restart if health check fails

### Performance Optimization
- Use `Standard` plan for better performance
- Consider `Pro` plan for production use
- Monitor CPU and memory usage

## Production Considerations

### Security
- Disable debug mode in production
- Use environment variables for sensitive data
- Enable HTTPS (automatic on Render)

### Performance
- Model files are cached in persistent disk
- First load may be slower due to model loading
- Consider warming up models on startup

### Scaling
- Render provides auto-scaling
- Monitor resource usage and upgrade plan if needed
- Consider horizontal scaling for high traffic

## Cost Optimization
- **Starter Plan**: $7/month (suitable for testing)
- **Standard Plan**: $25/month (recommended for production)
- **Pro Plan**: $85/month (high performance)

### Storage Costs
- 1GB persistent disk: Free
- Additional storage: $0.25/GB/month

## Common Issues & Solutions

### Build Timeout
If build takes too long:
1. Use the optimized `requirements.render.txt`
2. Enable build caching
3. Remove unnecessary dependencies

### Memory Issues
If app crashes due to memory:
1. Upgrade to Standard or Pro plan
2. Optimize model loading
3. Add memory monitoring

### Missing Model Files
If models not found:
1. Ensure all .pkl files are committed to Git
2. Check file paths in applications
3. Verify disk mount path `/app/data`

### Port Issues
If app doesn't start:
1. Ensure using `$PORT` environment variable
2. Check Dockerfile CMD uses dynamic port
3. Verify health check endpoint

## Accessing Your Application

Once deployed, your app will be available at:
```
https://molecular-prediction-suite.onrender.com
```

## Updating Your Application

To update:
1. Push changes to your Git repository
2. Render will automatically redeploy
3. Monitor logs for successful deployment

## Support

For issues:
- Check Render documentation: https://render.com/docs
- Review application logs in Render dashboard
- Verify all model files are present
