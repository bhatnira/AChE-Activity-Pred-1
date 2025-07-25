#!/bin/bash

# Startup script for Molecular Prediction Suite Docker container

echo "Starting virtual display..."
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
export DISPLAY=:99

# Wait a moment for Xvfb to start
sleep 2

# Set the port (default to 8501 if not provided by Render)
export PORT=${PORT:-8501}

echo "Starting Streamlit app on port $PORT..."

# Start Streamlit
exec streamlit run app_launcher.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
