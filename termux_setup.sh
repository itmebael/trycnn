#!/data/data/com.termux/files/usr/bin/bash
# Termux Setup Script for Flask Server on Android
# Run this script in Termux after installing Termux from Play Store

echo "=========================================="
echo "  Flask Server Setup for Android (Termux)"
echo "=========================================="
echo ""

# Step 1: Update packages
echo "[1/6] Updating Termux packages..."
pkg update -y && pkg upgrade -y

# Step 2: Install Python
echo ""
echo "[2/6] Installing Python..."
pkg install python -y

# Step 3: Install required system packages
echo ""
echo "[3/6] Installing system dependencies..."
pkg install git wget curl -y

# Step 4: Install Python packages
echo ""
echo "[4/6] Installing Python packages..."
pip install --upgrade pip
pip install flask werkzeug pillow numpy requests python-dotenv

# Step 5: Create project directory
echo ""
echo "[5/6] Setting up project directory..."
cd ~
mkdir -p trycnn
cd trycnn

# Step 6: Instructions
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=========================================="
echo "  NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. Copy your project files to:"
echo "   ~/trycnn/"
echo ""
echo "2. You can copy files using:"
echo "   - termux-setup-storage (to access Downloads)"
echo "   - git clone (if using git)"
echo "   - Manual copy via file manager"
echo ""
echo "3. Navigate to project:"
echo "   cd ~/trycnn"
echo ""
echo "4. Run Flask server:"
echo "   python app.py"
echo ""
echo "5. Access from browser:"
echo "   http://localhost:5000"
echo ""
echo "=========================================="



