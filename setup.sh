#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         🎵 LOFI GENERATOR - SETUP SCRIPT 🎵              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

echo "✅ Python $(python3 --version) detected"
echo "✅ Node.js $(node --version) detected"
echo ""

# Backend setup
echo "📦 Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "   Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "   Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

echo "   Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✅ Backend dependencies installed"
else
    echo "   ❌ Failed to install backend dependencies"
    exit 1
fi

cd ..

# Frontend setup
echo ""
echo "📦 Setting up frontend..."
cd frontend

echo "   Installing Node.js dependencies..."
npm install --silent

if [ $? -eq 0 ]; then
    echo "   ✅ Frontend dependencies installed"
else
    echo "   ❌ Failed to install frontend dependencies"
    exit 1
fi

cd ..

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p backend/models backend/uploads backend/outputs
mkdir -p data/images data/midi
echo "   ✅ Directories created"

# Final message
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                  ✅ SETUP COMPLETE!                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Add your training data:"
echo "   - Place images in: data/images/"
echo "   - Place MIDI files in: data/midi/"
echo ""
echo "2. Edit train_model.py to add your training pairs"
echo ""
echo "3. Train the model:"
echo "   python train_model.py"
echo ""
echo "4. Start the backend (in one terminal):"
echo "   cd backend"
echo "   source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
echo "   python app.py"
echo ""
echo "5. Start the frontend (in another terminal):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "6. Open http://localhost:3000 in your browser"
echo ""
echo "🎵 Happy music generating!"
echo ""
