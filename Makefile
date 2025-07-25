# Makefile for Molecular Prediction Suite Docker operations

.PHONY: help build up down logs test clean restart dev health

# Default target
help:
	@echo "Molecular Prediction Suite - Docker Commands"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build    - Build the Docker image"
	@echo "  make up       - Start the application"
	@echo "  make down     - Stop the application"
	@echo "  make restart  - Restart the application"
	@echo "  make logs     - View application logs"
	@echo "  make test     - Run Docker setup tests"
	@echo "  make dev      - Start in development mode"
	@echo "  make health   - Check application health"
	@echo "  make clean    - Clean up containers and images"
	@echo ""
	@echo "Quick start: make up"

# Build the Docker image
build:
	@echo "🔨 Building Docker image..."
	docker-compose build

# Start the application
up:
	@echo "🚀 Starting Molecular Prediction Suite..."
	docker-compose up -d
	@echo "✅ Application started!"
	@echo "🌐 Open http://localhost:8501 in your browser"

# Stop the application
down:
	@echo "🛑 Stopping application..."
	docker-compose down
	@echo "✅ Application stopped"

# Restart the application
restart: down up

# View logs
logs:
	@echo "📋 Viewing application logs..."
	docker-compose logs -f molecular-prediction-suite

# Run tests
test:
	@echo "🧪 Running Docker setup tests..."
	./test-docker.sh

# Start in development mode
dev:
	@echo "🛠️ Starting in development mode..."
	docker-compose -f docker-compose.dev.yml up
	@echo "🌐 Development server at http://localhost:8501"

# Check application health
health:
	@echo "🏥 Checking application health..."
	@if curl -f http://localhost:8501/_stcore/health 2>/dev/null; then \
		echo "✅ Application is healthy"; \
	else \
		echo "❌ Application is not responding"; \
	fi

# Clean up containers and images
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker image prune -f
	@echo "✅ Cleanup complete"

# Force rebuild and start
rebuild: clean build up

# Show container status
status:
	@echo "📊 Container status:"
	docker-compose ps
