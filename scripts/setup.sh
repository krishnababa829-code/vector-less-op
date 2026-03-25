#!/bin/bash
# Zero-Null Vectorless RAG - Setup Script

set -e

echo "=== Zero-Null Vectorless RAG Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    echo "Error: Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package
echo "Installing dependencies..."
pip install -e ".[dev]"

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install chromium

# Create data directories
mkdir -p data/raw data/markdown data/index

# Copy env template
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from template"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Start llama.cpp server:"
echo "   ./llama-server -m qwen2.5-3b-instruct.gguf --port 8000"
echo ""
echo "2. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run pipeline:"
echo "   vnull pipeline https://example.com --name my-docs"
echo ""
echo "4. Start API server:"
echo "   vnull serve"
