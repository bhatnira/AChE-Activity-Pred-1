#!/bin/bash

# Startup script for Molecular Prediction Suite Docker container

echo "🚀 Starting Molecular Prediction Suite..."

# Check if all required model files exist
echo "📋 Checking model files..."

# Create data directory if it doesn't exist
mkdir -p /app/data

# Check for required pickle files
if [ ! -f "best_model_aggregrate_circular.pkl" ]; then
    echo "⚠️  Warning: best_model_aggregrate_circular.pkl not found"
fi

if [ ! -f "bestPipeline_tpot_circularfingerprint_classification.pkl" ]; then
    echo "⚠️  Warning: bestPipeline_tpot_circularfingerprint_classification.pkl not found"
fi

if [ ! -f "bestPipeline_tpot_rdkit_classification.pkl" ]; then
    echo "⚠️  Warning: bestPipeline_tpot_rdkit_classification.pkl not found"
fi

if [ ! -f "train_data.pkl" ]; then
    echo "⚠️  Warning: train_data.pkl not found"
fi

# Check for checkpoint directory
if [ ! -d "checkpoint-2000" ]; then
    echo "⚠️  Warning: checkpoint-2000 directory not found"
fi

# Check for graph model directories
if [ ! -d "GraphConv_model_files" ]; then
    echo "⚠️  Warning: GraphConv_model_files directory not found"
fi

echo "🧬 Starting Streamlit app launcher..."

# Start the application
exec streamlit run app_launcher.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
