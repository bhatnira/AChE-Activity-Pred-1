#!/bin/bash

# Render.com startup script for Molecular Prediction Suite

echo "🚀 Starting Molecular Prediction Suite on Render..."

# Set default PORT if not provided by Render
export PORT=${PORT:-10000}

echo "📋 Environment Information:"
echo "PORT: $PORT"
echo "NODE: $(uname -n)"
echo "Python: $(python --version)"
echo "Working Directory: $(pwd)"

# Check if all required model files exist
echo "📋 Checking model files..."

if [ ! -f "best_model_aggregrate_circular.pkl" ]; then
    echo "⚠️  Warning: best_model_aggregrate_circular.pkl not found"
fi

if [ ! -f "bestPipeline_tpot_circularfingerprint_classification.pkl" ]; then
    echo "⚠️  Warning: bestPipeline_tpot_circularfingerprint_classification.pkl not found"
fi

if [ ! -f "bestPipeline_tpot_rdkit_classification.pkl" ]; then
    echo "⚠️  Warning: bestPipeline_tpot_rdkit_classification.pkl not found"
fi

# Create data directory if it doesn't exist
mkdir -p /app/data

# Set Streamlit configuration
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE=light
export STREAMLIT_THEME_PRIMARY_COLOR="#007AFF"

echo "🧬 Starting Streamlit app on port $PORT..."

# Start the application
exec streamlit run app_launcher.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base=light \
    --theme.primaryColor="#007AFF"
