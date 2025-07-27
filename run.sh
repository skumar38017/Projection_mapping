#!/bin/bash

# Random Object Verification System - Quick Start Script

echo "🎯 Random Object Verification System"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "backend/start_server.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: backend/start_server.py"
    exit 1
fi

# Change to backend directory
cd backend

echo "📦 Checking Python environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d "../venv" ]; then
    echo "💡 No virtual environment found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "🔄 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    echo "💡 Try running: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies installed"

# Run system test
echo "🧪 Running system tests..."
python test_system.py

if [ $? -ne 0 ]; then
    echo "⚠️  Some tests failed, but you can still try running the server"
    echo "🤔 Do you want to continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping. Please fix the issues and try again."
        exit 1
    fi
fi

# Check for reference images
if [ ! "$(ls -A app/assets/*.{jpg,jpeg,png} 2>/dev/null)" ]; then
    echo "⚠️  No reference images found in app/assets/"
    echo "📁 Please add some .jpg, .jpeg, or .png files to app/assets/ directory"
    echo "🤔 Do you want to continue without reference images? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "🛑 Please add reference images and try again."
        exit 1
    fi
fi

# Start the server
echo ""
echo "🚀 Starting the Random Object Verification System..."
echo "📱 Open your browser and go to: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python start_server.py
