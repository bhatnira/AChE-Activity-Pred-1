# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Expose Streamlit port (Render uses dynamic PORT)
EXPOSE $PORT

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check using dynamic port
HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health

# Command to run the app launcher with dynamic port
CMD ["sh", "-c", "streamlit run app_launcher.py --server.port=$PORT --server.address=0.0.0.0"]
