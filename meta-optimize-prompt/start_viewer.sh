#!/bin/bash

# MetaPromptBench Startup Script

echo "🚀 Starting MetaPromptBench..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Error: Python is not installed or not in PATH"
        echo "Please install Python 3.6+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "📁 Working directory: $SCRIPT_DIR"
echo "🐍 Using Python: $PYTHON_CMD"
echo ""

# Start the server
echo "🌐 Starting server on http://localhost:8000"
echo "📊 Open http://localhost:8000/evaluation_viewer.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"

cd "$SCRIPT_DIR"
$PYTHON_CMD server.py
