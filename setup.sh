#!/bin/bash
# Quick Start Script for Rainwater Harvesting Simulator

echo "🌧️  Rainwater Harvesting Simulation Platform - Setup"
echo "======================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
python --version
echo ""

# Create virtual environment (optional)
echo "Would you like to create a virtual environment? (recommended)"
echo "1. Yes"
echo "2. No (use global Python)"
read -p "Choose (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    
    echo "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    echo "✓ Virtual environment activated"
else
    echo "Skipping virtual environment creation"
fi

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "======================================================"
echo "✓ Setup complete!"
echo ""
echo "To run the application, execute:"
echo "  streamlit run app.py"
echo ""
echo "The app will open in your browser at:"
echo "  http://localhost:8501"
echo ""
