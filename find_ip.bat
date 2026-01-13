@echo off
echo ========================================
echo   FINDING YOUR COMPUTER'S IP ADDRESS
echo ========================================
echo.
echo This is the IP address you need to use
echo on your Android phone to access the app:
echo.
ipconfig | findstr /i "IPv4"
echo.
echo ========================================
echo.
echo Copy the IP address above (e.g., 192.168.1.100)
echo Then on your Android phone, open browser and go to:
echo http://YOUR_IP_ADDRESS:5000
echo.
echo Example: http://192.168.1.100:5000
echo.
pause



