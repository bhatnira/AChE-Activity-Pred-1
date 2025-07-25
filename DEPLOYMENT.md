# Deployment Guide for Render.com

This guide helps you deploy the Molecular Prediction Suite on Render.com using Docker and GitHub.

## Prerequisites

1. GitHub account
2. Render.com account
3. Your code repository on GitHub

## Deployment Steps

### 1. Prepare Your GitHub Repository

1. Push all your code to a GitHub repository
2. Ensure these files are included:
   - `app_launcher.py` (main application)
   - `Dockerfile.launcher` (Docker configuration)
   - `requirements.txt` (Python dependencies)
   - `render.yaml` (Render configuration)
   - All model files and apps (`app_*.py`)

### 2. Connect to Render.com

1. Go to [render.com](https://render.com)
2. Sign up/login with your GitHub account
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository

### 3. Configure Deployment

#### Option A: Using render.yaml (Recommended)
1. Render will automatically detect the `render.yaml` file
2. Review the configuration and click "Apply"

#### Option B: Manual Configuration
If you prefer manual setup:
- **Name**: `molecular-prediction-suite`
- **Environment**: `Docker`
- **Dockerfile Path**: `./Dockerfile.launcher`
- **Region**: Choose your preferred region
- **Instance Type**: `Starter` (or higher for better performance)

### 4. Environment Variables

The following environment variables are automatically set:
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

### 5. Deploy

1. Click "Create Web Service"
2. Render will:
   - Pull your code from GitHub
   - Build the Docker image
   - Deploy your application
   - Provide you with a live URL

## Post-Deployment

### Accessing Your App
- Your app will be available at: `https://your-service-name.onrender.com`
- Initial deployment may take 10-15 minutes

### Monitoring
- Check the deployment logs in Render dashboard
- Monitor app performance and usage
- Set up health checks (already configured)

## Performance Optimization

### For Better Performance:
1. Upgrade to a paid plan for faster builds and more resources
2. Consider using Render's Redis for caching
3. Enable autoscaling if needed

### Cost Considerations:
- Free tier: Limited hours per month
- Starter plan: $7/month for always-on service
- Standard plan: $25/month for better performance

## Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check requirements.txt for incompatible versions
   - Ensure all files are committed to GitHub
   - Review build logs in Render dashboard

2. **App Won't Start**
   - Verify Dockerfile.launcher syntax
   - Check if all required files are present
   - Review runtime logs

3. **Import Errors**
   - Ensure all dependencies are in requirements.txt
   - Check for missing model files

### Support Resources:
- [Render Documentation](https://render.com/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy)
- Check Render community forums

## Security Notes

- Model files and sensitive data are included in the build
- Consider using environment variables for sensitive configurations
- Render provides HTTPS by default

## Updates

To update your deployed app:
1. Push changes to your GitHub repository
2. Render will automatically rebuild and deploy
3. Enable auto-deploy for seamless updates

## File Structure

Your repository should have this structure:
```
your-repo/
├── app_launcher.py          # Main application
├── app_chemberta.py         # ChemBERTa app
├── app_rdkit.py            # RDKit app
├── app_circular.py         # Circular fingerprints app
├── app_graph_combined.py   # Graph neural networks app
├── Dockerfile.launcher     # Docker configuration
├── requirements.txt        # Python dependencies
├── render.yaml            # Render configuration
├── .dockerignore          # Docker ignore file
├── README.md              # This file
└── models/                # Model files
    ├── *.pkl
    ├── *.bin
    └── checkpoint*/
```

## Next Steps

1. Test your deployed application thoroughly
2. Set up monitoring and alerts
3. Consider implementing CI/CD for automated deployments
4. Monitor usage and scale as needed
