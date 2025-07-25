# Docker Setup Guide for Molecular Prediction Suite

## Overview
This Docker setup containerizes the Molecular Prediction Suite, which consists of a main launcher (`app_launcher.py`) that coordinates four different molecular prediction applications:

1. **ChemBERTa Transformer** - State-of-the-art transformer model for molecular property prediction
2. **RDKit Descriptors** - AutoML TPOT-based predictions using RDKit descriptors
3. **Circular Fingerprints** - Ensemble activity prediction using circular fingerprints
4. **Graph Neural Networks** - Deep learning on molecular graph representations

## Prerequisites
- Docker Desktop installed and running
- Docker Compose (usually included with Docker Desktop)
- At least 4GB of available RAM
- All model files and checkpoint directories present in the project folder

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up --build

# Access the application
# Open your browser and go to: http://localhost:8501
```

### Option 2: Using Docker directly
```bash
# Build the image
docker build -t molecular-prediction-suite .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  molecular-prediction-suite
```

## Container Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT=8501` - Port for Streamlit application
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0` - Allow external connections
- `STREAMLIT_SERVER_HEADLESS=true` - Run without browser
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` - Disable telemetry

### Volumes
- `./data:/app/data` - Persistent data storage
- `./models:/app/models` - Model files storage

### Ports
- `8501:8501` - Streamlit web interface

## Required Files
Ensure these files are present in the project directory:
- `app_launcher.py` (main coordinator)
- `app_chemberta.py`
- `app_rdkit.py` 
- `app_circular.py`
- `app_graph_combined.py`
- Model files (*.pkl)
- Checkpoint directories
- `requirements.txt`

## Docker Commands

### Build and Run
```bash
# Build the image
docker-compose build

# Start the services
docker-compose up

# Start in detached mode (background)
docker-compose up -d

# View logs
docker-compose logs -f
```

### Management
```bash
# Stop the services
docker-compose down

# Remove containers and networks
docker-compose down --remove-orphans

# Remove everything including volumes
docker-compose down -v

# Restart services
docker-compose restart
```

### Debugging
```bash
# Enter the running container
docker-compose exec molecular-prediction-suite bash

# View container logs
docker-compose logs molecular-prediction-suite

# Check container status
docker-compose ps
```

## Troubleshooting

### Common Issues

1. **Port 8501 already in use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8502:8501"  # Use port 8502 instead
   ```

2. **Out of memory errors**
   - Increase Docker Desktop memory allocation to at least 4GB
   - Close other resource-intensive applications

3. **Model files not found**
   - Ensure all .pkl files and checkpoint directories are present
   - Check file permissions

4. **Container won't start**
   ```bash
   # Check logs for specific errors
   docker-compose logs
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Health Check
The container includes a health check that verifies Streamlit is running:
```bash
# Check health status
docker-compose ps
```

## Development Mode

For development with live code changes:
```bash
# Mount the current directory
docker run -p 8501:8501 \
  -v $(pwd):/app \
  -it molecular-prediction-suite \
  bash
```

## Production Deployment

For production deployment, consider:
1. Using a reverse proxy (nginx)
2. Setting up SSL certificates
3. Using Docker secrets for sensitive data
4. Implementing proper logging
5. Setting up monitoring

## Performance Tips

1. **Memory**: Allocate at least 4GB RAM to Docker
2. **Storage**: Use SSD for better I/O performance
3. **Network**: Ensure stable internet for model downloads
4. **CPU**: Multi-core processors recommended for parallel processing

## Security Considerations

1. Don't expose sensitive model files in public images
2. Use Docker secrets for API keys
3. Run containers with non-root users when possible
4. Keep base images updated

## Backup and Recovery

```bash
# Backup data volume
docker run --rm -v $(pwd)/data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore data volume
docker run --rm -v $(pwd)/data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data
```
