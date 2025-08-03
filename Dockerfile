# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including X11 libraries for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgconf-2-4 \
    libxss1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libx11-6 \
    libx11-dev \
    libxext6 \
    libsm6 \
    libice6 \
    xvfb \
    libcairo2-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV MPLBACKEND=Agg
ENV QT_QPA_PLATFORM=offscreen
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Make startup script executable (check if it exists first)
RUN if [ -f /app/start.sh ]; then chmod +x /app/start.sh; else echo "start.sh not found, will use direct command"; fi

# Default port for local development, will be overridden by Render
ENV PORT=8501

# Expose the port
EXPOSE $PORT

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check using curl with port fallback
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || curl --fail http://localhost:8501/_stcore/health

# Use the startup script if available, otherwise direct command
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & export DISPLAY=:99 && streamlit run app_launcher.py --server.port=10000 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=true --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
