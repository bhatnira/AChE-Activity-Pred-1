# Molecular Prediction Suite - Docker Setup

A containerized molecular prediction platform that coordinates four different AI-powered applications for acetylcholinesterase inhibitor prediction.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM allocated to Docker
- All model files present in the project directory

### Run the Application
```bash
# Start the application
docker-compose up

# Access the web interface
open http://localhost:8501
```

## 📋 Application Structure

The `app_launcher.py` serves as the main coordinator for four specialized prediction applications:

### 🧪 ChemBERTa Transformer
- **File**: `app_chemberta.py`
- **Features**: State-of-the-art transformer model with attention visualization
- **Capabilities**: SMILES input, molecular drawing, batch processing

### ⚛️ RDKit Descriptors  
- **File**: `app_rdkit.py`
- **Features**: AutoML TPOT-based predictions using RDKit molecular descriptors
- **Capabilities**: Automated feature selection, optimized ML pipelines

### 🔄 Circular Fingerprints
- **File**: `app_circular.py` 
- **Features**: Ensemble modeling using Morgan/ECFP fingerprints
- **Capabilities**: Molecular similarity analysis, activity prediction

### 🕸️ Graph Neural Networks
- **File**: `app_graph_combined.py`
- **Features**: Deep learning on molecular graph representations
- **Capabilities**: Graph convolution, node/edge feature learning

## 🐳 Docker Configuration

### Container Architecture
```
┌─────────────────────────────────────┐
│           app_launcher.py           │
│         (Main Coordinator)          │
├─────────────────────────────────────┤
│  🧪 ChemBERTa │ ⚛️ RDKit │ 🔄 Circular │
│  🕸️ Graph NN  │           │            │
├─────────────────────────────────────┤
│      Python 3.10 + Dependencies    │
├─────────────────────────────────────┤
│         Docker Container            │
└─────────────────────────────────────┘
```

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 🛠️ Setup Options

### Option 1: Production Setup
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Option 2: Development Setup
```bash
# Use development configuration with live reloading
docker-compose -f docker-compose.dev.yml up

# The source code is mounted as a volume for live changes
```

### Option 3: Manual Docker Commands
```bash
# Build image
docker build -t molecular-prediction-suite .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  molecular-prediction-suite
```

## 📁 Required Files Checklist

### Core Application Files
- ✅ `app_launcher.py` - Main coordinator
- ✅ `app_chemberta.py` - ChemBERTa application
- ✅ `app_rdkit.py` - RDKit application  
- ✅ `app_circular.py` - Circular fingerprints application
- ✅ `app_graph_combined.py` - Graph neural networks application

### Model Files
- ✅ `best_model_aggregrate_circular.pkl`
- ✅ `bestPipeline_tpot_circularfingerprint_classification.pkl`
- ✅ `bestPipeline_tpot_rdkit_classification.pkl`
- ✅ `train_data.pkl`

### Model Directories
- ✅ `checkpoint-2000/` - ChemBERTa model checkpoint
- ✅ `GraphConv_model_files/` - Graph convolution models
- ✅ `graphConv_reg_model_files 2/` - Graph regression models

### Configuration Files
- ✅ `requirements.txt` - Python dependencies
- ✅ `Dockerfile` - Container definition
- ✅ `docker-compose.yml` - Multi-container setup

## 🔧 Management Commands

### Container Operations
```bash
# Start services
docker-compose up

# Stop services  
docker-compose down

# Restart services
docker-compose restart

# View status
docker-compose ps

# View logs
docker-compose logs molecular-prediction-suite
```

### Debugging
```bash
# Enter running container
docker-compose exec molecular-prediction-suite bash

# Check health
curl http://localhost:8501/_stcore/health

# View real-time logs
docker-compose logs -f
```

### Cleanup
```bash
# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker image prune -f
```

## 🧪 Testing

Run the automated test script:
```bash
./test-docker.sh
```

This will verify:
- Docker is running
- Required files are present
- Container builds successfully
- Application starts and responds

## 🌐 Accessing the Application

Once running, access the application at:
- **URL**: http://localhost:8501
- **Health Check**: http://localhost:8501/_stcore/health

The main interface provides cards for each application with launch buttons.

## 🐛 Troubleshooting

### Common Issues

**Port 8501 in use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"
```

**Out of memory:**
- Increase Docker Desktop memory allocation
- Close other applications

**Model files missing:**
```bash
# Check required files
ls -la *.pkl
ls -la checkpoint-2000/
ls -la GraphConv_model_files/
```

**Container won't start:**
```bash
# Check detailed logs
docker-compose logs

# Rebuild without cache
docker-compose build --no-cache
```

## 📊 Performance Tips

1. **Memory**: Allocate 4GB+ RAM to Docker Desktop
2. **Storage**: Use SSD for model file I/O
3. **CPU**: Multi-core recommended for ML inference
4. **Network**: Stable connection for dependency downloads

## 🔒 Security Notes

- Container runs as non-root user
- No sensitive data in image layers
- Health checks enabled
- Minimal attack surface

## 📝 Development

For development with live code changes:
```bash
# Use development compose file
docker-compose -f docker-compose.dev.yml up

# Code changes will be reflected immediately
# due to volume mounting and file watching
```

## 🚀 Production Deployment

For production environments:
1. Use a reverse proxy (nginx/traefik)
2. Set up SSL/TLS certificates
3. Configure proper logging
4. Implement monitoring
5. Use Docker secrets for sensitive data
6. Set resource limits

## 📞 Support

If you encounter issues:
1. Run `./test-docker.sh` to diagnose problems
2. Check logs with `docker-compose logs`
3. Verify all model files are present
4. Ensure Docker has sufficient resources allocated
