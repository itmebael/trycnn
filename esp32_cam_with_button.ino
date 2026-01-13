#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiClient.h>

// -------- Wi-Fi & Server --------
// IMPORTANT: Update these with your phone hotspot name and password
const char* WIFI_SSID = "YOUR_PHONE_HOTSPOT_NAME";     // CHANGE: Your phone hotspot name
const char* WIFI_PASS = "YOUR_HOTSPOT_PASSWORD";       // CHANGE: Your phone hotspot password

// IMPORTANT: Find your laptop's IP when connected to hotspot (use ipconfig in CMD)
// Steps:
// 1. Connect laptop to phone hotspot
// 2. Run "ipconfig" in CMD
// 3. Find your IPv4 Address (usually 192.168.43.xxx for Android or 172.20.10.xxx for iPhone)
// 4. Replace the IP below with YOUR laptop's IP
const char* SERVER_HOST = "192.168.43.XXX";              // CHANGE: Your laptop IP on hotspot (use ipconfig to find it!)
const int   SERVER_PORT = 5000;
const char* SERVER_PATH = "/api/predict";

// HTTP Server for streaming
WebServer server(80);

// -------- Video tuning --------
// Target ~640p-equivalent: use VGA (640x480). Higher loads may reduce FPS.
const framesize_t STREAM_FRAME_SIZE = FRAMESIZE_VGA; // 640x480
// JPEG quality: lower number = better quality, larger size. 18 is a balanced default.
const int JPEG_QUALITY = 18;
// Aim for ~20-25 fps for stability at this resolution/bandwidth.
const uint16_t STREAM_FRAME_DELAY_MS = 50;           // ~20 fps target

// -------- Camera pins (AI Thinker) --------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void connectWiFi() {
  WiFi.mode(WIFI_STA); // Set WiFi to station mode
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  
  Serial.print("WiFi connecting to: ");
  Serial.println(WIFI_SSID);
  Serial.print("Connecting");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) { 
    delay(500); 
    Serial.print("."); 
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n========================================");
    Serial.println("WiFi connected successfully!");
    Serial.println("========================================");
    Serial.print("ESP32-CAM IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    Serial.println("========================================");
    Serial.print("Camera Stream URL: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/stream");
    Serial.print("Camera Capture URL: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/capture");
    Serial.println("========================================\n");
  } else {
    Serial.println("\nWiFi connection failed!");
    Serial.print("Status: ");
    Serial.println(WiFi.status());
    Serial.println("Check your WiFi SSID and password.");
    // Keep trying to connect
    while (WiFi.status() != WL_CONNECTED) {
      delay(5000);
      Serial.println("Retrying WiFi connection...");
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = STREAM_FRAME_SIZE; // Tunable resolution
  config.jpeg_quality = JPEG_QUALITY;      // Tunable compression
  config.fb_count     = 1;

  // Try to initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }
  
  // Give camera time to stabilize
  delay(1000);
  Serial.println("Camera initialized successfully");
  return true;
}

void postImage() {
  // Try multiple times to get frame
  camera_fb_t *fb = NULL;
  for (int i = 0; i < 3; i++) {
    fb = esp_camera_fb_get();
    if (fb) break;
    delay(100);
    Serial.println("Retrying camera capture...");
  }
  
  if (!fb) { 
    Serial.println("Capture failed after retries"); 
    return; 
  }
  
  Serial.printf("Captured image: %d bytes\n", fb->len);

  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("Connection failed");
    esp_camera_fb_return(fb);
    return;
  }

  String boundary = "----esp32boundary";
  String head = "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"leafImage\"; filename=\"capture.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";
  size_t contentLength = head.length() + fb->len + tail.length();

  // HTTP request
  client.print(String("POST ") + SERVER_PATH + " HTTP/1.1\r\n");
  client.print(String("Host: ") + SERVER_HOST + ":" + SERVER_PORT + "\r\n");
  client.print("Content-Type: multipart/form-data; boundary=" + boundary + "\r\n");
  client.print("Content-Length: " + String(contentLength) + "\r\n");
  client.print("Connection: close\r\n\r\n");

  client.print(head);
  client.write(fb->buf, fb->len);
  client.print(tail);

  // Read response
  unsigned long start = millis();
  while (client.connected() && (millis() - start < 5000)) {
    while (client.available()) {
      String line = client.readStringUntil('\n');
      Serial.println(line);
      start = millis();
    }
  }
  client.stop();
  esp_camera_fb_return(fb);
}

// Handle MJPEG stream
void handleStream() {
  WiFiClient client = server.client();
  
  // Send HTTP headers
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.println();

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      break;
    }

    // Send frame boundary and headers
    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.print("Content-Length: ");
    client.print(fb->len);
    client.print("\r\n\r\n");
    
    // Send image data
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    
    esp_camera_fb_return(fb);
    delay(STREAM_FRAME_DELAY_MS); // Adjust frame rate
  }
  
  client.stop();
}

// Handle single image capture
void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: image/jpeg");
  client.println("Content-Disposition: inline; filename=capture.jpg");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Content-Length: " + String(fb->len));
  client.println("Connection: close");
  client.println();
  client.write(fb->buf, fb->len);
  client.stop();
  
  esp_camera_fb_return(fb);
}

// Handle root page
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>ESP32-CAM</title></head><body>";
  html += "<h1>ESP32-CAM Stream</h1>";
  html += "<p>Stream URL: <a href='/stream'>/stream</a></p>";
  html += "<p>Capture URL: <a href='/capture'>/capture</a></p>";
  html += "<img src='/stream' style='max-width:100%;'/>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void setupServer() {
  server.on("/", handleRoot);
  server.on("/stream", handleStream);
  server.on("/capture", handleCapture);
  server.begin();
  Serial.println("========================================");
  Serial.println("HTTP server started on port 80");
  Serial.println("========================================");
  Serial.print("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Stream URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
  Serial.print("Capture URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capture");
  Serial.println("========================================\n");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n========================================");
  Serial.println("ESP32-CAM Starting...");
  Serial.println("========================================\n");
  
  connectWiFi();
  initCamera();
  // Warm-up period to let exposure/white balance settle for better quality
  delay(10000); // 10 seconds
  setupServer();
  
  Serial.println("========================================");
  Serial.println("System Ready!");
  Serial.println("========================================");
  Serial.print("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Use this IP in the dashboard to view camera");
  Serial.println("========================================\n");
}

void loop() {
  // Handle HTTP server requests
  server.handleClient();
  
  // Print IP address reminder every 30 seconds
  static unsigned long lastIPReminder = 0;
  if (millis() - lastIPReminder > 30000) {
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n--- ESP32-CAM IP Address: " + WiFi.localIP().toString() + " ---");
      Serial.println("Stream: http://" + WiFi.localIP().toString() + "/stream");
    }
    lastIPReminder = millis();
  }

  delay(10); // Small delay for server handling
}

