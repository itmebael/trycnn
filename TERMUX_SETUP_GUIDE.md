# Running Flask Server on Android using Termux

## Quick Start Guide

### Step 1: Install Termux
1. Open **Google Play Store** on your Android device
2. Search for **"Termux"**
3. Install **Termux** (by Fredrik Fornwall)
4. Open Termux app

### Step 2: Run Setup Script
Copy the `termux_setup.sh` file to your Android device, then:

```bash
# In Termux, navigate to where you saved the script
cd ~/storage/downloads  # or wherever you saved it

# Make it executable
chmod +x termux_setup.sh

# Run the setup
./termux_setup.sh
```

**OR** run commands manually:

```bash
# Update packages
pkg update && pkg upgrade

# Install Python
pkg install python

# Install Python packages
pip install flask werkzeug pillow numpy requests python-dotenv
```

### Step 3: Copy Project Files to Android

**Option A: Using Termux Storage Access**
```bash
# Grant storage access
termux-setup-storage

# Copy files from Downloads
cp -r ~/storage/downloads/trycnn ~/trycnn
```

**Option B: Using Git (if project is on GitHub)**
```bash
cd ~
git clone YOUR_GITHUB_REPO_URL trycnn
cd trycnn
```

**Option C: Using USB/File Manager**
1. Connect Android to computer via USB
2. Copy entire `trycnn` folder to Android's Download folder
3. In Termux:
   ```bash
   termux-setup-storage
   cp -r ~/storage/downloads/trycnn ~/trycnn
   ```

### Step 4: Install Project Dependencies
```bash
cd ~/trycnn

# Install all requirements
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install flask werkzeug pillow numpy requests python-dotenv supabase
```

### Step 5: Configure Environment (if needed)
```bash
# Create .env file if needed
nano .env
# Add your environment variables
# Press Ctrl+X, then Y, then Enter to save
```

### Step 6: Run Flask Server
```bash
cd ~/trycnn
python app.py
```

You should see:
```
Running on http://0.0.0.0:5000
```

### Step 7: Access the App

**On the same Android device:**
- Open Chrome or any browser
- Go to: `http://localhost:5000`

**From other devices on same WiFi:**
1. Find Android's IP address:
   ```bash
   ifconfig | grep "inet "
   ```
2. On other device, go to: `http://ANDROID_IP:5000`

---

## Important Notes

### Keep Termux Running
- **Don't close Termux** - it will stop the server
- Use **Termux:Widget** or **Termux:Boot** to auto-start
- Or use **Termux:API** for background tasks

### Battery Optimization
- Disable battery optimization for Termux
- Settings → Apps → Termux → Battery → Unrestricted

### Network Access
- Make sure Android firewall allows Termux
- Check if port 5000 is accessible

### Storage Permissions
- Run `termux-setup-storage` to access files
- This creates a symlink to `/sdcard`

---

## Troubleshooting

### "Command not found: python"
```bash
pkg install python
```

### "Module not found"
```bash
pip install MODULE_NAME
```

### "Permission denied"
```bash
chmod +x termux_setup.sh
```

### Can't access from other devices
- Check Android's IP: `ifconfig`
- Make sure both devices on same WiFi
- Check Android firewall settings

### Server stops when closing Termux
- Use `nohup` to run in background:
  ```bash
  nohup python app.py &
  ```
- Or use Termux:Boot for auto-start

---

## Advanced: Run in Background

```bash
# Run in background
nohup python app.py > flask.log 2>&1 &

# Check if running
ps aux | grep python

# View logs
tail -f flask.log

# Stop server
pkill -f "python app.py"
```

---

## Alternative: Pydroid 3 (Easier GUI)

If Termux is too complex, try **Pydroid 3**:
1. Install from Play Store
2. Open app.py in Pydroid
3. Install packages via Pip tab
4. Run the script
5. Access via `http://localhost:5000`



