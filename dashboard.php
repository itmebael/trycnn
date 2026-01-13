<?php
session_start();
if (!isset($_SESSION["user"])) {
    header("Location: login.php");
    exit();
}

// Handle file upload
$upload_status = '';
$detection_result = '';

// Handle file deletion
if (isset($_GET['delete']) && !empty($_GET['delete'])) {
    $file_to_delete = $_GET['delete'];
    $file_path = "uploads/" . basename($file_to_delete);
    
    if (file_exists($file_path)) {
        if (unlink($file_path)) {
            $upload_status = "File deleted successfully.";
            
            // Also remove from detection results
            $results_file = "uploads/detection_results.json";
            if (file_exists($results_file)) {
                $results = json_decode(file_get_contents($results_file), true) ?: [];
                $filename = basename($file_to_delete);
                if (isset($results[$filename])) {
                    unset($results[$filename]);
                    file_put_contents($results_file, json_encode($results, JSON_PRETTY_PRINT));
                }
            }
        } else {
            $upload_status = "Error deleting file.";
        }
    } else {
        $upload_status = "File not found.";
    }
    
    // Redirect to results page after deletion
    header("Location: ?page=results");
    exit();
}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_FILES['leafImage'])) {
    $target_dir = "uploads/";
    if (!file_exists($target_dir)) {
        mkdir($target_dir, 0777, true);
    }
    
    $target_file = $target_dir . basename($_FILES["leafImage"]["name"]);
    $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));
    
    // Check if file is an image
    if (getimagesize($_FILES["leafImage"]["tmp_name"]) !== false) {
        if (move_uploaded_file($_FILES["leafImage"]["tmp_name"], $target_file)) {
            $upload_status = "File uploaded successfully.";
            
            // Simulate detection result (replace with actual CNN model)
            $detection_result = simulateDetection($target_file);
            
            // Store detection result in a JSON file for persistence
            $results_file = "uploads/detection_results.json";
            $results = [];
            
            if (file_exists($results_file)) {
                $results = json_decode(file_get_contents($results_file), true) ?: [];
            }
            
            $filename = basename($target_file);
            $results[$filename] = $detection_result;
            
            file_put_contents($results_file, json_encode($results, JSON_PRETTY_PRINT));
        } else {
            $upload_status = "Error uploading file.";
        }
    } else {
        $upload_status = "File is not an image.";
    }
}

function simulateDetection($image_path) {
    // Simulate detection results - replace with actual CNN model
    $conditions = ['Healthy', 'Diseased'];
    $confidence = rand(75, 95);
    $condition = $conditions[array_rand($conditions)];
    
    // Generate recommendations based on condition
    $recommendations = getRecommendations($condition);
    
    return [
        'condition' => $condition,
        'confidence' => $confidence,
        'timestamp' => date('Y-m-d H:i:s'),
        'image_path' => $image_path,
        'recommendations' => $recommendations
    ];
}

function getRecommendations($condition) {
    if ($condition === 'Healthy') {
        return [
            'title' => '🌱 Your pechay leaf is healthy!',
            'tips' => [
                'Continue current care routine',
                'Maintain proper watering schedule',
                'Ensure adequate sunlight (6-8 hours daily)',
                'Monitor for pests regularly',
                'Apply organic fertilizer monthly',
                'Keep soil well-drained'
            ],
            'action' => 'Keep up the excellent work! Your pechay is thriving.'
        ];
    } else {
        return [
            'title' => '⚠️ Your pechay leaf shows signs of disease',
            'tips' => [
                'Check soil moisture - avoid overwatering',
                'Improve air circulation around plants',
                'Remove affected leaves immediately',
                'Apply fungicide if fungal infection suspected',
                'Ensure proper drainage',
                'Consider organic treatments first'
            ],
            'action' => 'Take immediate action to prevent spread to other plants.'
        ];
    }
}

// Get current page
$page = isset($_GET['page']) ? $_GET['page'] : 'dashboard';

// Calculate dashboard statistics from actual detection results
function getDashboardStats() {
    global $BASE_DIR;
    $uploads_dir = $BASE_DIR . "/uploads/";
    $results_file = $uploads_dir . "detection_results.json";
    $stats = [
        'total_scans' => 0,
        'healthy_leaves' => 0,
        'diseased_leaves' => 0,
        'success_rate' => 0
    ];
    
    if (is_dir($uploads_dir)) {
        $files = scandir($uploads_dir);
        $image_files = array_filter($files, function($file) {
            $ext = strtolower(pathinfo($file, PATHINFO_EXTENSION));
            return in_array($ext, ['jpg', 'jpeg', 'png', 'gif']);
        });
        
        $stats['total_scans'] = count($image_files);
        
        // Load stored detection results
        $stored_results = [];
        if (file_exists($results_file)) {
            $stored_results = json_decode(file_get_contents($results_file), true) ?: [];
        }
        
        // Count actual detection results
        foreach ($image_files as $file) {
            if (isset($stored_results[$file])) {
                $condition = $stored_results[$file]['condition'];
                if ($condition === 'Healthy') {
                    $stats['healthy_leaves']++;
                } else {
                    $stats['diseased_leaves']++;
                }
            } else {
                // For files without stored results, simulate and store
                $condition = ['Healthy', 'Diseased'][array_rand(['Healthy', 'Diseased'])];
                if ($condition === 'Healthy') {
                    $stats['healthy_leaves']++;
                } else {
                    $stats['diseased_leaves']++;
                }
                
                // Store the simulated result
                $stored_results[$file] = [
                    'condition' => $condition,
                    'confidence' => rand(75, 95),
                    'timestamp' => date('Y-m-d H:i:s'),
                    'image_path' => 'uploads/' . $file,
                    'recommendations' => getRecommendations($condition)
                ];
            }
        }
        
        // Update stored results if we added new ones
        if (count($stored_results) > 0) {
            file_put_contents($results_file, json_encode($stored_results, JSON_PRETTY_PRINT));
        }
        
        // Calculate success rate based on confidence scores
        if ($stats['total_scans'] > 0) {
            $total_confidence = 0;
            $confidence_count = 0;
            
            foreach ($stored_results as $result) {
                if (isset($result['confidence'])) {
                    $total_confidence += $result['confidence'];
                    $confidence_count++;
                }
            }
            
            if ($confidence_count > 0) {
                $stats['success_rate'] = round($total_confidence / $confidence_count);
            } else {
                $stats['success_rate'] = rand(75, 95);
            }
        }
    }
    
    return $stats;
}

$dashboard_stats = getDashboardStats();
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pechay Leaf Detection Dashboard</title>
  <style>
    body {
      margin: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f4f8f4;
    }
    .sidebar {
      position: fixed;
      left: 0;
      top: 0;
      width: 220px;
      height: 100%;
      background: #2e7d32;
      color: #fff;
      padding-top: 20px;
      box-shadow: 2px 0 6px rgba(0,0,0,0.2);
    }
    .sidebar h2 { text-align: center; margin-bottom: 30px; font-size: 20px; }
    .sidebar a {
      display: block; color: #fff; padding: 12px 20px;
      text-decoration: none; font-size: 16px; transition: 0.3s;
    }
    .sidebar a:hover { background: #1b5e20; padding-left: 25px; }
    .sidebar a.active { background: #1b5e20; font-weight: bold; }
    .main { margin-left: 220px; }
    .header {
      background: #388e3c; color: #fff; padding: 15px 25px;
      font-size: 20px; font-weight: bold;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .content {
      padding: 30px;
      display: block;
      min-height: 70vh;
    }
    .card {
      background: #fff; padding: 25px; border-radius: 15px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      max-width: 650px; width: 100%; text-align: center;
    }
    .card.upload-page {
      max-width: 1200px;
      text-align: left;
      margin: 20px auto;
    }
    h1 { color: #2e7d32; margin-bottom: 10px; }
    p { color: #555; margin-bottom: 25px; }
    input[type="file"] {
      margin: 15px 0; padding: 10px;
      border: 2px solid #2e7d32; border-radius: 10px;
      cursor: pointer; width: 100%;
    }
    button {
      background: #2e7d32; color: white; padding: 14px 28px;
      border: none; border-radius: 12px; font-size: 16px;
      font-weight: bold; cursor: pointer; transition: 0.3s;
      margin-top: 15px;
    }
    button:hover { background: #1b5e20; transform: scale(1.05); }
    img.preview { max-width: 100%; border-radius: 12px; margin-top: 20px; display: none; }
    footer {
      text-align: center; margin: 30px 0 0; padding: 15px;
      background: #f1f1f1; font-size: 14px; color: #777;
    }
    .status-message {
      padding: 10px; margin: 10px 0; border-radius: 5px;
      background: #d4edda; color: #155724; border: 1px solid #c3e6cb;
    }
    .error-message {
      background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
    }
    .result-card {
      background: #fff; padding: 20px; margin: 20px 0; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #2e7d32;
    }
    .stats-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px; margin: 20px 0;
    }
    .stat-card {
      background: #fff; padding: 20px; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
    }
    .stat-number { font-size: 2em; font-weight: bold; color: #2e7d32; }
    .stat-label { color: #666; margin-top: 5px; }
    .delete-btn {
      background: #dc3545; color: white; padding: 8px 16px;
      border: none; border-radius: 6px; font-size: 14px;
      cursor: pointer; margin-top: 10px; transition: 0.3s;
    }
    .delete-btn:hover { background: #c82333; transform: scale(1.05); }
    .result-actions { margin-top: 15px; }
    .recommendations {
      background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 10px;
      border-left: 4px solid #28a745; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .recommendations.diseased { border-left-color: #dc3545; }
    .recommendations h4 { color: #2e7d32; margin-bottom: 15px; font-size: 1.1rem; }
    .recommendations.diseased h4 { color: #dc3545; }
    .recommendations ul { margin: 10px 0; padding-left: 20px; }
    .recommendations li { margin: 8px 0; color: #555; }
    .recommendations .action { background: #e8f5e8; padding: 12px; border-radius: 8px; margin-top: 15px; font-weight: 500; color: #2e7d32; }
    .recommendations.diseased .action { background: #f8e8e8; color: #dc3545; }
    .camera-container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin: 20px 0;
    }
    .camera-section {
      background: #fff !important;
      padding: 25px !important;
      border-radius: 15px !important;
      border: 3px solid #2e7d32 !important;
      box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2) !important;
      display: block !important;
      visibility: visible !important;
      margin-top: 30px !important;
      width: 100% !important;
      box-sizing: border-box !important;
    }
    .upload-section {
      background: #f8f9fa;
      padding: 20px;
      border-radius: 10px;
      border: 2px solid #e0e0e0;
    }
    .camera-section h2, .camera-section h3, .upload-section h3 {
      color: #2e7d32;
      margin-top: 0;
    }
    #cameraStream {
      width: 100%;
      max-width: 100%;
      border-radius: 8px;
      border: 2px solid #2e7d32;
      background: #000;
      min-height: 300px;
      object-fit: contain;
    }
    .camera-controls {
      margin-top: 15px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .camera-controls input {
      flex: 2;
      min-width: 200px;
      padding: 12px;
      border: 2px solid #2e7d32;
      border-radius: 8px;
      font-size: 1rem;
    }
    .camera-controls button {
      padding: 12px 20px;
      margin: 0;
      border: none;
      border-radius: 8px;
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s;
    }
    .camera-controls button:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stream-status {
      padding: 10px;
      margin: 10px 0;
      border-radius: 5px;
      background: #fff3cd;
      color: #856404;
      border: 1px solid #ffeaa7;
    }
    .stream-status.connected {
      background: #d4edda;
      color: #155724;
      border: 1px solid #c3e6cb;
    }
    .stream-status.error {
      background: #f8d7da;
      color: #721c24;
      border: 1px solid #f5c6cb;
    }
    @media (max-width: 768px) {
      .camera-container {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>

  <div class="sidebar">
    <h2>🌱 Pechay System</h2>
    <a href="?page=dashboard" class="<?php echo $page == 'dashboard' ? 'active' : ''; ?>">📊 Dashboard</a>
    <a href="?page=upload" class="<?php echo $page == 'upload' ? 'active' : ''; ?>">📷 Upload Leaf</a>
    <a href="?page=results" class="<?php echo $page == 'results' ? 'active' : ''; ?>">📝 Results</a>
    <a href="logout.php">🚪 Logout</a>
  </div>

  <div class="main">
    <div class="header">Pechay Leaf Detection Dashboard</div>
    <div class="content">
      <?php if ($page == 'dashboard'): ?>
        <div class="card">
          <h1>📊 Dashboard Overview</h1>
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-number"><?php echo $dashboard_stats['total_scans']; ?></div>
              <div class="stat-label">Total Scans</div>
            </div>
            <div class="stat-card">
              <div class="stat-number"><?php echo $dashboard_stats['healthy_leaves']; ?></div>
              <div class="stat-label">Healthy Leaves</div>
            </div>
            <div class="stat-card">
              <div class="stat-number"><?php echo $dashboard_stats['diseased_leaves']; ?></div>
              <div class="stat-label">Diseased Leaves</div>
            </div>
            <div class="stat-card">
              <div class="stat-number"><?php echo $dashboard_stats['success_rate']; ?>%</div>
              <div class="stat-label">Success Rate</div>
            </div>
          </div>
          
          <?php if ($dashboard_stats['total_scans'] == 0): ?>
            <div class="status-message">
              <h3>🌱 Welcome to Pechay Detection System!</h3>
              <p>No scans yet. Upload your first pechay leaf image to get started with detection.</p>
            </div>
          <?php else: ?>
            <div class="status-message">
              <h3>📈 Recent Activity</h3>
              <p>You have processed <?php echo $dashboard_stats['total_scans']; ?> leaf images.</p>
              <p><strong>Health Distribution:</strong> <?php echo round(($dashboard_stats['healthy_leaves'] / $dashboard_stats['total_scans']) * 100); ?>% healthy, <?php echo round(($dashboard_stats['diseased_leaves'] / $dashboard_stats['total_scans']) * 100); ?>% diseased</p>
            </div>
          <?php endif; ?>
          
          <div style="margin-top: 25px;">
            <a href="?page=upload" style="display: inline-block; background: #2e7d32; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; margin-right: 15px;">📷 Upload New Image</a>
            <a href="?page=results" style="display: inline-block; background: #388e3c; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px;">📝 View Results</a>
          </div>
        </div>
      
      <?php elseif ($page == 'upload'): ?>
        <!-- ESP32-CAM Live Stream Section - MUST BE VISIBLE -->
        <div class="camera-section" id="esp32CameraSection" style="background: #fff !important; border: 4px solid #2e7d32 !important; box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3) !important; display: block !important; visibility: visible !important; position: relative !important; z-index: 100 !important; padding: 30px !important; border-radius: 15px !important; margin: 20px auto !important; max-width: 1200px !important; width: 95% !important;">
          <h2 style="color: #2e7d32; margin-top: 0; margin-bottom: 20px; font-size: 1.8rem; font-weight: bold; display: block !important; text-align: center !important;">📹 ESP32-CAM Live View</h2>
          <p style="color: #666; margin-bottom: 20px; display: block !important; text-align: center !important; font-size: 1.1rem;">Connect to your ESP32-CAM to view live feed</p>
            <div id="streamStatus" class="stream-status">
              Enter ESP32-CAM IP address and click "Start Stream" or "Start Polling"
            </div>
            <div class="camera-controls">
              <input type="text" id="cameraIP" placeholder="ESP32-CAM IP (e.g., 192.168.1.100)" value="" onkeypress="if(event.key==='Enter') startStream()" style="flex: 2; min-width: 250px;">
              <button type="button" onclick="startStream()" style="background: #2e7d32;">▶️ Start Stream</button>
              <button type="button" onclick="startPolling()" style="background: #17a2b8;">🔄 Start Polling</button>
              <button type="button" onclick="stopStream()" style="background: #dc3545;">⏹️ Stop</button>
              <button type="button" onclick="testConnection()" style="background: #6c757d;">🔍 Test</button>
            </div>
            <div id="cameraViewContainer" style="position: relative; margin-top: 20px; background: #000; border-radius: 12px; min-height: 450px; max-height: 700px; display: flex; align-items: center; justify-content: center; border: 3px solid #2e7d32; overflow: hidden; width: 100%;">
              <img id="cameraStream" src="" alt="Camera stream will appear here" style="display: none; max-width: 100%; max-height: 100%; width: auto; height: auto; border-radius: 8px; object-fit: contain;">
              <iframe id="cameraStreamFrame" src="" style="display: none; width: 100%; height: 100%; min-height: 450px; border: none; border-radius: 8px;"></iframe>
              <canvas id="captureCanvas" style="display: none;"></canvas>
              <div id="streamPlaceholder" style="color: #999; text-align: center; padding: 40px; width: 100%;">
                <div style="font-size: 4rem; margin-bottom: 20px;">📷</div>
                <div style="font-size: 1.3rem; margin-bottom: 15px; color: #fff;">Camera Live View</div>
                <div style="color: #aaa; margin-bottom: 10px;">Enter ESP32-CAM IP address above</div>
                <small style="color: #888;">Click "Start Polling" or "Start Stream" to begin</small>
              </div>
            </div>
            <div class="camera-controls" style="margin-top: 20px; justify-content: center;">
              <button type="button" onclick="captureFromStream()" id="captureBtn" disabled style="background: #28a745; padding: 15px 40px; font-size: 1.1rem; font-weight: bold;">📸 Capture & Detect Leaf Condition</button>
            </div>
        </div>
        
        <div class="card upload-page" style="margin-top: 30px;">
          <h1>📷 Upload & Detect</h1>
          <p>Upload a pechay leaf image (from ESP32-CAM or computer) to check condition.</p>
          
          <!-- File Upload Section - Alternative Option -->
          <div class="upload-section" style="margin-top: 20px;">
            <h3>📁 Alternative: Upload from Computer</h3>
            <form method="POST" enctype="multipart/form-data" id="uploadForm">
              <input type="file" name="leafImage" accept="image/*" onchange="previewImage(event)" required>
              <br>
              <img id="preview" class="preview">
              <br>
              <button type="submit">Detect Leaf Condition</button>
            </form>
          </div>
          
          <?php if ($upload_status): ?>
            <div class="status-message <?php echo strpos($upload_status, 'Error') !== false ? 'error-message' : ''; ?>" style="margin-top: 30px;">
              <?php echo $upload_status; ?>
            </div>
          <?php endif; ?>
          
          <?php if ($detection_result): ?>
            <div class="result-card" style="margin-top: 30px;">
              <h3>🔍 Detection Result</h3>
              <p><strong>Condition:</strong> <?php echo $detection_result['condition']; ?></p>
              <p><strong>Confidence:</strong> <?php echo $detection_result['confidence']; ?>%</p>
              <p><strong>Timestamp:</strong> <?php echo $detection_result['timestamp']; ?></p>
              <?php if (file_exists($detection_result['image_path'])): ?>
                <img src="<?php echo $detection_result['image_path']; ?>" style="max-width: 300px; border-radius: 8px; margin-top: 15px;">
              <?php endif; ?>
            </div>
            
            <?php if (isset($detection_result['recommendations'])): ?>
              <div class="recommendations <?php echo strtolower($detection_result['condition']); ?>" style="margin-top: 20px;">
                <h4><?php echo $detection_result['recommendations']['title']; ?></h4>
                <ul>
                  <?php foreach ($detection_result['recommendations']['tips'] as $tip): ?>
                    <li><?php echo $tip; ?></li>
                  <?php endforeach; ?>
                </ul>
                <div class="action"><?php echo $detection_result['recommendations']['action']; ?></div>
              </div>
            <?php endif; ?>
          <?php endif; ?>
        </div>
      
      <?php elseif ($page == 'results'): ?>
        <div class="card">
          <h1>📝 Detection Results</h1>
          <p>Recent leaf detection results and analysis.</p>
          
          <?php
          // Get uploaded files for results display with accurate detection results
          $uploads_dir = "uploads/";
          $results = [];
          $results_file = "uploads/detection_results.json";
          $stored_results = [];
          
          if (file_exists($results_file)) {
            $stored_results = json_decode(file_get_contents($results_file), true) ?: [];
          }
          
          if (is_dir($uploads_dir)) {
            $files = scandir($uploads_dir);
            foreach ($files as $file) {
              if ($file != '.' && $file != '..' && $file != 'detection_results.json' && in_array(strtolower(pathinfo($file, PATHINFO_EXTENSION)), ['jpg', 'jpeg', 'png', 'gif'])) {
                $fpath = $uploads_dir . $file;
                
                // Use stored result if available, otherwise simulate
                if (isset($stored_results[$file])) {
                  $result = $stored_results[$file];
                } else {
                  // Simulate for existing files without stored results
                  $result = simulateDetection($fpath);
                  $stored_results[$file] = $result;
                }
                
                $results[] = [
                    'filename' => $file,
                    'path' => $uploads_dir . $file,
                    'timestamp' => date('Y-m-d H:i:s', filemtime($fpath)),
                    'condition' => $result['condition'],
                    'confidence' => $result['confidence'],
                    'recommendations' => isset($result['recommendations']) ? $result['recommendations'] : getRecommendations($result['condition'])
                ];
              }
            }
            
            // Update stored results if we added new ones
            if (count($stored_results) > 0) {
              file_put_contents($results_file, json_encode($stored_results, JSON_PRETTY_PRINT));
            }
          }
          
          if (empty($results)): ?>
            <div class="status-message">
              No detection results found. Upload some images first!
            </div>
          <?php else: ?>
            <?php foreach ($results as $result): ?>
              <div class="result-card">
                <h4><?php echo $result['filename']; ?></h4>
                <p><strong>Condition:</strong> <?php echo $result['condition']; ?></p>
                <p><strong>Confidence:</strong> <?php echo $result['confidence']; ?>%</p>
                <p><strong>Date:</strong> <?php echo $result['timestamp']; ?></p>
                <img src="<?php echo $result['path']; ?>" style="max-width: 200px; border-radius: 8px; margin-top: 10px;">
                
                <?php if (isset($result['recommendations'])): ?>
                  <div class="recommendations <?php echo strtolower($result['condition']); ?>">
                    <h4><?php echo $result['recommendations']['title']; ?></h4>
                    <ul>
                      <?php foreach ($result['recommendations']['tips'] as $tip): ?>
                        <li><?php echo $tip; ?></li>
                      <?php endforeach; ?>
                    </ul>
                    <div class="action"><?php echo $result['recommendations']['action']; ?></div>
                  </div>
                <?php endif; ?>
                
                <div class="result-actions">
                  <button class="delete-btn" onclick="confirmDelete('<?php echo $result['filename']; ?>')">
                    🗑️ Remove
                  </button>
                </div>
              </div>
            <?php endforeach; ?>
          <?php endif; ?>
        </div>
      <?php endif; ?>
    </div>
    <footer>
      &copy; 2025 Pechay Detection System | Group 4 BSIT 4A
    </footer>
  </div>

  <script>
    let streamInterval = null;
    let cameraStreamImg = null;
    
    // Debug: Check if camera section is loaded
    window.addEventListener('DOMContentLoaded', function() {
      const cameraSection = document.querySelector('.camera-section');
      const cameraIP = document.getElementById('cameraIP');
      const esp32Section = document.getElementById('esp32CameraSection');
      
      console.log('=== Camera Section Debug ===');
      if (cameraSection) {
        console.log('✓ Camera section found');
        console.log('Display:', window.getComputedStyle(cameraSection).display);
        console.log('Visibility:', window.getComputedStyle(cameraSection).visibility);
        // Force visibility
        cameraSection.style.display = 'block';
        cameraSection.style.visibility = 'visible';
      } else {
        console.error('✗ Camera section NOT found!');
      }
      if (esp32Section) {
        console.log('✓ ESP32 Camera Section ID found');
        esp32Section.style.display = 'block';
        esp32Section.style.visibility = 'visible';
      }
      if (cameraIP) {
        console.log('✓ Camera IP input found');
      } else {
        console.error('✗ Camera IP input NOT found!');
      }
      console.log('===========================');
    });
    
    function previewImage(event) {
      var reader = new FileReader();
      reader.onload = function(){
        var output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block';
      };
      reader.readAsDataURL(event.target.files[0]);
    }
    
    function confirmDelete(filename) {
      if (confirm('Are you sure you want to delete "' + filename + '"? This action cannot be undone.')) {
        window.location.href = '?page=results&delete=' + encodeURIComponent(filename);
      }
    }
    
    function startStream() {
      const cameraIP = document.getElementById('cameraIP').value.trim();
      const statusDiv = document.getElementById('streamStatus');
      const streamImg = document.getElementById('cameraStream');
      const captureBtn = document.getElementById('captureBtn');
      const placeholder = document.getElementById('streamPlaceholder');
      
      if (!cameraIP) {
        statusDiv.className = 'stream-status error';
        statusDiv.textContent = 'Please enter ESP32-CAM IP address';
        return;
      }
      
      // Stop any existing stream
      stopStream();
      
      // Hide placeholder
      if (placeholder) placeholder.style.display = 'none';
      
      // Construct stream URL (ESP32-CAM typically uses /stream endpoint)
      const streamUrl = 'http://' + cameraIP + '/stream';
      
      // Update status
      statusDiv.className = 'stream-status';
      statusDiv.textContent = 'Connecting to camera at ' + cameraIP + '...';
      
      // Set up image element
      streamImg.style.display = 'block';
      streamImg.style.width = '100%';
      streamImg.style.height = 'auto';
      cameraStreamImg = streamImg;
      
      // Also try iframe method as fallback
      const streamFrame = document.getElementById('cameraStreamFrame');
      streamFrame.style.display = 'none';
      
      // Add timestamp to prevent caching
      const timestamp = new Date().getTime();
      streamImg.src = streamUrl + '?t=' + timestamp;
      
      // Try iframe method after a delay if img doesn't work
      setTimeout(function() {
        if (statusDiv.textContent.includes('Connecting') || statusDiv.textContent.includes('Still connecting')) {
          streamFrame.src = streamUrl;
          streamFrame.style.display = 'block';
          streamImg.style.display = 'none';
          statusDiv.textContent = 'Using iframe stream method...';
        }
      }, 3000);
      
      // Check if stream is loading
      let loadTimeout = setTimeout(function() {
        if (statusDiv.textContent.includes('Connecting')) {
          statusDiv.className = 'stream-status';
          statusDiv.textContent = 'Still connecting... Trying alternative methods...';
          tryAlternativeStream(cameraIP, 0);
        }
      }, 2000);
      
      streamImg.onload = function() {
        clearTimeout(loadTimeout);
        statusDiv.className = 'stream-status connected';
        statusDiv.textContent = '✓ Stream connected successfully!';
        captureBtn.disabled = false;
        if (placeholder) placeholder.style.display = 'none';
      };
      
      streamImg.onerror = function() {
        clearTimeout(loadTimeout);
        if (statusDiv.textContent.includes('Connecting') || statusDiv.textContent.includes('Still connecting')) {
          // Try alternative endpoints
          tryAlternativeStream(cameraIP, 0);
        } else {
          statusDiv.className = 'stream-status error';
          statusDiv.textContent = '✗ Failed to connect. Check IP address and ensure ESP32-CAM is streaming.';
          streamImg.style.display = 'none';
          captureBtn.disabled = true;
          if (placeholder) placeholder.style.display = 'block';
        }
      };
    }
    
    function tryAlternativeStream(cameraIP, index) {
      const altUrls = [
        'http://' + cameraIP + '/stream',
        'http://' + cameraIP + '/mjpeg/stream',
        'http://' + cameraIP + '/cam.mjpeg',
        'http://' + cameraIP + '/capture'
      ];
      
      if (index >= altUrls.length) {
        const statusDiv = document.getElementById('streamStatus');
        const streamImg = document.getElementById('cameraStream');
        const placeholder = document.getElementById('streamPlaceholder');
        statusDiv.className = 'stream-status error';
        statusDiv.textContent = '✗ Failed to connect. Make sure ESP32-CAM is running and accessible at ' + cameraIP;
        streamImg.style.display = 'none';
        if (placeholder) placeholder.style.display = 'block';
        return;
      }
      
      const streamImg = document.getElementById('cameraStream');
      const statusDiv = document.getElementById('streamStatus');
      const placeholder = document.getElementById('streamPlaceholder');
      
      statusDiv.textContent = 'Trying: ' + altUrls[index] + '...';
      const timestamp = new Date().getTime();
      streamImg.src = altUrls[index] + (altUrls[index].includes('?') ? '&' : '?') + 't=' + timestamp;
      
      streamImg.onload = function() {
        statusDiv.className = 'stream-status connected';
        statusDiv.textContent = '✓ Stream connected successfully!';
        document.getElementById('captureBtn').disabled = false;
        if (placeholder) placeholder.style.display = 'none';
      };
      
      streamImg.onerror = function() {
        setTimeout(function() {
          tryAlternativeStream(cameraIP, index + 1);
        }, 1000);
      };
    }
    
    
    function stopStream() {
      const streamImg = document.getElementById('cameraStream');
      const streamFrame = document.getElementById('cameraStreamFrame');
      const statusDiv = document.getElementById('streamStatus');
      const captureBtn = document.getElementById('captureBtn');
      const placeholder = document.getElementById('streamPlaceholder');
      
      if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
      }
      
      // Stop the stream by setting src to empty
      streamImg.src = '';
      streamImg.style.display = 'none';
      streamFrame.src = '';
      streamFrame.style.display = 'none';
      statusDiv.className = 'stream-status';
      statusDiv.textContent = 'Stream stopped';
      captureBtn.disabled = true;
      cameraStreamImg = null;
      if (placeholder) placeholder.style.display = 'block';
    }
    
    function captureFromStream() {
      const streamImg = document.getElementById('cameraStream');
      const canvas = document.getElementById('captureCanvas');
      const statusDiv = document.getElementById('streamStatus');
      
      if (!streamImg || !streamImg.src || streamImg.style.display === 'none') {
        alert('Please start the camera stream first');
        return;
      }
      
      try {
        // Set canvas dimensions to match stream
        canvas.width = streamImg.naturalWidth || streamImg.width || 640;
        canvas.height = streamImg.naturalHeight || streamImg.height || 480;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(streamImg, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob and upload
        canvas.toBlob(function(blob) {
          const formData = new FormData();
          const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
          formData.append('leafImage', blob, 'esp32_capture_' + timestamp + '.jpg');
          
          statusDiv.className = 'stream-status';
          statusDiv.textContent = 'Uploading captured image...';
          
          // Submit to the same upload endpoint
          fetch(window.location.href, {
            method: 'POST',
            body: formData
          })
          .then(response => {
            if (response.ok) {
              statusDiv.className = 'stream-status connected';
              statusDiv.textContent = '✓ Image captured and uploaded successfully!';
              // Reload page to show detection result
              setTimeout(function() {
                window.location.reload();
              }, 1500);
            } else {
              throw new Error('Upload failed');
            }
          })
          .catch(error => {
            statusDiv.className = 'stream-status error';
            statusDiv.textContent = '✗ Failed to upload image: ' + error.message;
          });
        }, 'image/jpeg', 0.95);
      } catch (error) {
        statusDiv.className = 'stream-status error';
        statusDiv.textContent = '✗ Capture failed: ' + error.message;
        console.error('Capture error:', error);
      }
    }
    
    function startPolling() {
      console.log('startPolling() called');
      const cameraIP = document.getElementById('cameraIP');
      const statusDiv = document.getElementById('streamStatus');
      const streamImg = document.getElementById('cameraStream');
      const captureBtn = document.getElementById('captureBtn');
      const placeholder = document.getElementById('streamPlaceholder');
      
      if (!cameraIP) {
        alert('Camera IP input field not found! Please refresh the page.');
        console.error('Camera IP input not found');
        return;
      }
      
      const ipValue = cameraIP.value.trim();
      
      if (!ipValue) {
        if (statusDiv) {
          statusDiv.className = 'stream-status error';
          statusDiv.textContent = 'Please enter ESP32-CAM IP address';
        } else {
          alert('Please enter ESP32-CAM IP address');
        }
        return;
      }
      
      console.log('Starting polling with IP:', ipValue);
      
      // Stop any existing stream
      stopStream();
      
      // Hide placeholder
      if (placeholder) placeholder.style.display = 'none';
      
      const captureUrl = 'http://' + ipValue + '/capture';
      console.log('Capture URL:', captureUrl);
      streamImg.style.display = 'block';
      streamImg.style.width = '100%';
      streamImg.style.height = 'auto';
      cameraStreamImg = streamImg;
      
      if (statusDiv) {
        statusDiv.className = 'stream-status';
        statusDiv.textContent = 'Polling camera at ' + ipValue + '...';
      }
      if (captureBtn) captureBtn.disabled = false;
      
      // Poll every 500ms for near real-time view
      streamInterval = setInterval(function() {
        const timestamp = new Date().getTime();
        streamImg.src = captureUrl + '?t=' + timestamp;
      }, 500);
      
      // Initial load
      const timestamp = new Date().getTime();
      streamImg.src = captureUrl + '?t=' + timestamp;
      
      streamImg.onload = function() {
        console.log('Image loaded successfully');
        if (statusDiv) {
          statusDiv.className = 'stream-status connected';
          statusDiv.textContent = '✓ Polling active - Camera connected!';
        }
      };
      
      streamImg.onerror = function() {
        console.error('Failed to load image from:', captureUrl);
        if (statusDiv) {
          statusDiv.className = 'stream-status error';
          statusDiv.textContent = '✗ Failed to connect. Check IP address and ensure ESP32-CAM is running.';
        }
        stopStream();
      };
    }
    
    function testConnection() {
      const cameraIP = document.getElementById('cameraIP').value.trim();
      const statusDiv = document.getElementById('streamStatus');
      
      if (!cameraIP) {
        statusDiv.className = 'stream-status error';
        statusDiv.textContent = 'Please enter ESP32-CAM IP address';
        return;
      }
      
      statusDiv.className = 'stream-status';
      statusDiv.textContent = 'Testing connection to ' + cameraIP + '...';
      
      // Test if we can reach the camera
      const testUrl = 'http://' + cameraIP + '/capture';
      const img = new Image();
      
      img.onload = function() {
        statusDiv.className = 'stream-status connected';
        statusDiv.textContent = '✓ Camera is reachable! You can now start the stream.';
      };
      
      img.onerror = function() {
        statusDiv.className = 'stream-status error';
        statusDiv.textContent = '✗ Cannot reach camera at ' + cameraIP + '. Check:\n- ESP32-CAM is powered on\n- Both devices are on same network\n- IP address is correct';
      };
      
      const timestamp = new Date().getTime();
      img.src = testUrl + '?t=' + timestamp;
    }
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
      stopStream();
    });
  </script>
</body>
</html>
