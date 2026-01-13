# Running Flask Server on Android

## Option 1: Access Flask Server from Android (Recommended)
If your Flask server is running on your computer, you can access it from your Android device:

### Steps:
1. **Make sure Flask server is running on your computer**
   ```cmd
   python app.py
   ```

2. **Find your computer's IP address:**
   - Windows: Open CMD and type `ipconfig`
   - Look for "IPv4 Address" (e.g., 192.168.1.100)

3. **On your Android device:**
   - Make sure Android and computer are on the same WiFi network
   - Open browser on Android
   - Go to: `http://YOUR_COMPUTER_IP:5000`
   - Example: `http://192.168.1.100:5000`

4. **Login and use the app normally**

---

## Option 2: Run Flask Server on Android Device (Advanced)

### Using Termux (Android Terminal Emulator)

1. **Install Termux from Google Play Store**

2. **Update packages:**
   ```bash
   pkg update && pkg upgrade
   ```

3. **Install Python:**
   ```bash
   pkg install python
   ```

4. **Install pip packages:**
   ```bash
   pip install flask
   pip install -r requirements.txt
   ```

5. **Clone or copy your project to Android:**
   ```bash
   cd ~
   # Copy files using termux-setup-storage
   termux-setup-storage
   # Then copy files from Downloads or use git clone
   ```

6. **Run Flask server:**
   ```bash
   cd ~/trycnn
   python app.py
   ```

7. **Access from Android browser:**
   - Open browser: `http://localhost:5000`
   - Or from other devices on same network: `http://ANDROID_IP:5000`

---

## Option 3: Using Pydroid 3 (Easier GUI)

1. **Install Pydroid 3 from Google Play Store**

2. **Open Pydroid 3**

3. **Install packages:**
   - Go to Pip tab
   - Install: flask, werkzeug, pillow, etc.

4. **Open your app.py file**

5. **Run the script**

---

## Option 4: Using Kivy/Python-for-Android (For Mobile App)

Convert your Flask web app into a native Android app using:
- **Buildozer** (Kivy)
- **BeeWare** (Briefcase)
- **Chaquopy** (Android Studio plugin)

---

## Quick Setup for Option 1 (Most Common)

### On Windows Computer:
1. Open CMD
2. Type: `ipconfig`
3. Note your IPv4 address (e.g., 192.168.1.100)
4. Run: `python app.py`

### On Android Device:
1. Connect to same WiFi as computer
2. Open Chrome/Browser
3. Go to: `http://192.168.1.100:5000`
4. Login and use the app!

---

## Troubleshooting:

**Can't access from Android?**
- Check firewall on Windows (allow port 5000)
- Make sure both devices on same WiFi
- Try using `0.0.0.0` instead of `127.0.0.1` in app.py (already set)

**Port blocked?**
- Windows Firewall: Allow Python through firewall
- Or run: `netsh advfirewall firewall add rule name="Flask" dir=in action=allow protocol=TCP localport=5000`



