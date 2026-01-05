#!/bin/bash

# Build and Test Script for calculus.py
# Usage: ./build.sh [clean|test|build|upload]

set -e

function clean() {
    echo "🧹 Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "✅ Clean complete!"
}

function test() {
    echo "🧪 Running tests..."
    echo "📦 Installing test dependencies..."
    pip install pytest
    python3 -m pytest test/ -v
    echo "✅ Tests complete!"
}

function lint() {
    echo "🔍 Running linting..."
    echo "📦 Installing linting dependencies..."
    pip install flake8 black
    python3 -m flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
    python3 -m black --check src/
    echo "✅ Linting complete!"
}

function build() {
    echo "🔨 Building package..."
    echo "📦 Installing build dependencies..."
    pip install --upgrade pip build twine
    clean
    python3 -m build
    python3 -m twine check dist/*
    echo "✅ Build complete!"
}

function upload_test() {
    echo "📦 Uploading to Test PyPI..."
    python3 -m twine upload --repository testpypi dist/*
    echo "✅ Upload to Test PyPI complete!"
}

function upload() {
    echo "📦 Uploading to PyPI..."
    python3 -m twine upload dist/*
    echo "✅ Upload to PyPI complete!"
}

function install_dev() {
    echo "💽 Installing in development mode..."
    pip install -e .[dev,jupyter]
    echo "✅ Development installation complete!"
}

# Main script logic
case "${1:-help}" in
    clean)
        clean
        ;;
    test)
        test
        ;;
    lint)
        lint
        ;;
    build)
        build
        ;;
    upload-test)
        upload_test
        ;;
    upload)
        upload
        ;;
    install-dev)
        install_dev
        ;;
    all)
        lint
        test
        build
        ;;
    help|*)
        echo "📚 Build script for calculus.py"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  clean      - Remove build artifacts"
        echo "  test       - Run test suite"
        echo "  lint       - Run code linting"
        echo "  build      - Build package"
        echo "  upload-test- Upload to Test PyPI"
        echo "  upload     - Upload to PyPI"
        echo "  install-dev- Install in development mode"
        echo "  all        - Run lint, test, and build"
        echo "  help       - Show this help message"
        ;;
esac
