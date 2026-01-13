<?php
session_start();

$mode = isset($_GET['mode']) ? $_GET['mode'] : 'login';
$error = '';
$success = '';

// Load users from JSON file
function loadUsers() {
    $file = 'users.json';
    if (file_exists($file)) {
        $data = file_get_contents($file);
        return json_decode($data, true) ?: [];
    }
    return [];
}

// Save users to JSON file
function saveUsers($users) {
    file_put_contents('users.json', json_encode($users, JSON_PRETTY_PRINT));
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_POST['action']) && $_POST['action'] === 'register') {
        // Registration handling
        $username = trim($_POST["username"] ?? '');
        $email = trim($_POST["email"] ?? '');
        $password = $_POST["password"] ?? '';
        $confirm_password = $_POST["confirm_password"] ?? '';
        
        // Validation
        if (empty($username) || empty($email) || empty($password) || empty($confirm_password)) {
            $error = "All fields are required!";
        } elseif (strlen($username) < 3) {
            $error = "Username must be at least 3 characters!";
        } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            $error = "Invalid email address!";
        } elseif (strlen($password) < 4) {
            $error = "Password must be at least 4 characters!";
        } elseif ($password !== $confirm_password) {
            $error = "Passwords do not match!";
        } else {
            $users = loadUsers();
            
            // Check if username or email already exists
            foreach ($users as $user) {
                if ($user['username'] === $username) {
                    $error = "Username already exists!";
                    break;
                }
                if ($user['email'] === $email) {
                    $error = "Email already registered!";
                    break;
                }
            }
            
            if (empty($error)) {
                // Add new user
                $users[] = [
                    'username' => $username,
                    'email' => $email,
                    'password' => password_hash($password, PASSWORD_DEFAULT),
                    'created_at' => date('Y-m-d H:i:s')
                ];
                saveUsers($users);
                $success = "Registration successful! You can now login.";
                $mode = 'login';
            }
        }
    } else {
        // Login handling
        $username = $_POST["username"] ?? '';
        $password = $_POST["password"] ?? '';
        
        // Check default admin account
        if ($username === "admin" && $password === "1234") {
            $_SESSION["user"] = $username;
            header("Location: dashboard.php");
            exit();
        }
        
        // Check registered users
        $users = loadUsers();
        $found = false;
        foreach ($users as $user) {
            if ($user['username'] === $username && password_verify($password, $user['password'])) {
                $_SESSION["user"] = $username;
                header("Location: dashboard.php");
                exit();
            }
        }
        
        if (!$found) {
            $error = "Invalid username or password!";
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login - Pechay Detection System</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
      background: linear-gradient(135deg, #2e7d32 0%, #388e3c 50%, #4caf50 100%);
      min-height: 100vh;
      display: flex; 
      justify-content: center; 
      align-items: center;
      position: relative;
      overflow: hidden;
    }
    
    body::before {
      content: '';
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/><circle cx="10" cy="60" r="0.5" fill="rgba(255,255,255,0.05)"/><circle cx="90" cy="40" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
      animation: float 20s ease-in-out infinite;
      z-index: 0;
    }
    
    @keyframes float {
      0%, 100% { transform: translateY(0px) rotate(0deg); }
      50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .login-container {
      position: relative;
      z-index: 1;
      width: 100%;
      max-width: 420px;
      padding: 20px;
    }
    
    .login-box { 
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      padding: 40px 35px; 
      border-radius: 20px; 
      box-shadow: 0 20px 40px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.2);
      text-align: center;
      border: 1px solid rgba(255,255,255,0.3);
      transition: all 0.3s ease;
    }
    
    .login-box:hover {
      transform: translateY(-5px);
      box-shadow: 0 25px 50px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.3);
    }
    
    .logo {
      font-size: 3.5rem;
      margin-bottom: 10px;
      animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
      0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
      40% { transform: translateY(-10px); }
      60% { transform: translateY(-5px); }
    }
    
    h1 { 
      color: #2e7d32; 
      font-size: 2.2rem;
      font-weight: 700;
      margin-bottom: 8px;
      text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
      color: #666;
      font-size: 1rem;
      margin-bottom: 35px;
      font-weight: 400;
    }
    
    .form-group {
      position: relative;
      margin-bottom: 25px;
    }
    
    .form-group i {
      position: absolute;
      left: 15px;
      top: 50%;
      transform: translateY(-50%);
      color: #2e7d32;
      font-size: 1.1rem;
    }
    
    input { 
      width: 100%; 
      padding: 15px 15px 15px 45px; 
      border: 2px solid #e0e0e0; 
      border-radius: 12px;
      font-size: 1rem;
      transition: all 0.3s ease;
      background: rgba(255,255,255,0.8);
    }
    
    input:focus {
      outline: none;
      border-color: #2e7d32;
      box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1);
      background: rgba(255,255,255,1);
    }
    
    input::placeholder {
      color: #999;
      font-weight: 400;
    }
    
    .login-btn { 
      background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%);
      color: white; 
      padding: 15px 30px; 
      width: 100%; 
      border: none; 
      border-radius: 12px; 
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: 600;
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    
    .login-btn::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
      transition: left 0.5s;
    }
    
    .login-btn:hover::before {
      left: 100%;
    }
    
    .login-btn:hover { 
      background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(46, 125, 50, 0.3);
    }
    
    .login-btn:active {
      transform: translateY(0);
    }
    
    .error { 
      color: #d32f2f; 
      margin-top: 15px; 
      padding: 12px;
      background: rgba(211, 47, 47, 0.1);
      border-radius: 8px;
      border-left: 4px solid #d32f2f;
      font-weight: 500;
      animation: shake 0.5s ease-in-out;
    }
    
    @keyframes shake {
      0%, 100% { transform: translateX(0); }
      25% { transform: translateX(-5px); }
      75% { transform: translateX(5px); }
    }
    
    .footer-text {
      margin-top: 30px;
      color: #666;
      font-size: 0.9rem;
    }
    
    .demo-credentials {
      background: rgba(46, 125, 50, 0.1);
      padding: 15px;
      border-radius: 10px;
      margin-top: 20px;
      border-left: 4px solid #2e7d32;
    }
    
    .demo-credentials h4 {
      color: #2e7d32;
      margin-bottom: 8px;
      font-size: 0.9rem;
    }
    
    .demo-credentials p {
      color: #666;
      font-size: 0.8rem;
      margin: 2px 0;
    }
    
    .success {
      color: #2e7d32;
      margin-top: 15px;
      padding: 12px;
      background: rgba(46, 125, 50, 0.1);
      border-radius: 8px;
      border-left: 4px solid #2e7d32;
      font-weight: 500;
      animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .toggle-mode {
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid #e0e0e0;
      text-align: center;
    }
    
    .toggle-mode p {
      margin: 0;
      color: #666;
      font-size: 0.95rem;
    }
    
    .toggle-mode a {
      color: #2e7d32;
      text-decoration: none;
      font-weight: 600;
      transition: all 0.3s ease;
      margin-left: 5px;
    }
    
    .toggle-mode a:hover {
      color: #1b5e20;
      text-decoration: underline;
    }
    
    .form-tabs {
      display: flex !important;
      gap: 10px;
      margin-bottom: 25px;
      border-bottom: 2px solid #e0e0e0;
      visibility: visible !important;
    }
    
    .form-tab {
      flex: 1;
      padding: 12px;
      text-align: center;
      background: transparent;
      border: none;
      border-bottom: 3px solid transparent;
      color: #666;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      display: block !important;
      visibility: visible !important;
    }
    
    .form-tab:hover {
      color: #2e7d32;
      background: rgba(46, 125, 50, 0.05);
    }
    
    .form-tab.active {
      color: #2e7d32;
      border-bottom-color: #2e7d32;
      background: rgba(46, 125, 50, 0.1);
    }
    
    .login-form {
      display: none;
    }
    
    .login-form.active {
      display: block;
    }
    
    .register-form {
      display: none;
    }
    
    .register-form.active {
      display: block;
    }
    
    @media (max-width: 480px) {
      .login-container {
        padding: 15px;
      }
      
      .login-box {
        padding: 30px 25px;
      }
      
      h1 {
        font-size: 1.8rem;
      }
      
      .logo {
        font-size: 3rem;
      }
    }
  </style>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
  <div class="login-container">
    <div class="login-box">
      <div class="logo">🌱</div>
      <h1>Pechay Detection System</h1>
      <p class="subtitle">BSIT 4A - Leaf Detection</p>
      
      <!-- Form Tabs -->
      <div class="form-tabs">
        <button type="button" class="form-tab <?php echo $mode === 'login' ? 'active' : ''; ?>" onclick="window.location.href='?mode=login'">
          <i class="fas fa-sign-in-alt"></i> Login
        </button>
        <button type="button" class="form-tab <?php echo $mode === 'register' ? 'active' : ''; ?>" onclick="window.location.href='?mode=register'">
          <i class="fas fa-user-plus"></i> Register
        </button>
      </div>
      
      <!-- Login Form -->
      <form method="POST" class="login-form <?php echo $mode === 'login' ? 'active' : ''; ?>" id="loginForm">
        <div class="form-group">
          <i class="fas fa-user"></i>
          <input type="text" name="username" placeholder="Enter your username" required value="<?php echo isset($_POST['username']) && $mode === 'login' ? htmlspecialchars($_POST['username']) : ''; ?>">
        </div>
        
        <div class="form-group">
          <i class="fas fa-lock"></i>
          <input type="password" name="password" placeholder="Enter your password" required>
        </div>
        
        <button type="submit" class="login-btn">
          <i class="fas fa-sign-in-alt"></i> Login to Dashboard
        </button>
        
        <?php if (!empty($error) && $mode === 'login'): ?>
          <div class="error">
            <i class="fas fa-exclamation-triangle"></i> <?php echo htmlspecialchars($error); ?>
          </div>
        <?php endif; ?>
        
        <?php if (!empty($success)): ?>
          <div class="success">
            <i class="fas fa-check-circle"></i> <?php echo htmlspecialchars($success); ?>
          </div>
        <?php endif; ?>
        
        <div style="margin-top: 20px; text-align: center;">
          <a href="?mode=register" style="display: inline-block; padding: 12px 24px; background: #fff; color: #2e7d32; border: 2px solid #2e7d32; border-radius: 12px; text-decoration: none; font-weight: 600; transition: all 0.3s ease;">
            <i class="fas fa-user-plus"></i> Create New Account
          </a>
        </div>
      </form>
      
      <!-- Register Form -->
      <form method="POST" class="register-form <?php echo $mode === 'register' ? 'active' : ''; ?>" id="registerForm">
        <input type="hidden" name="action" value="register">
        
        <div class="form-group">
          <i class="fas fa-user"></i>
          <input type="text" name="username" placeholder="Choose a username" required value="<?php echo isset($_POST['username']) && $mode === 'register' ? htmlspecialchars($_POST['username']) : ''; ?>">
        </div>
        
        <div class="form-group">
          <i class="fas fa-envelope"></i>
          <input type="email" name="email" placeholder="Enter your email" required value="<?php echo isset($_POST['email']) && $mode === 'register' ? htmlspecialchars($_POST['email']) : ''; ?>">
        </div>
        
        <div class="form-group">
          <i class="fas fa-lock"></i>
          <input type="password" name="password" placeholder="Create a password" required>
        </div>
        
        <div class="form-group">
          <i class="fas fa-lock"></i>
          <input type="password" name="confirm_password" placeholder="Confirm your password" required>
        </div>
        
        <button type="submit" class="login-btn">
          <i class="fas fa-user-plus"></i> Create Account
        </button>
        
        <?php if (!empty($error) && $mode === 'register'): ?>
          <div class="error">
            <i class="fas fa-exclamation-triangle"></i> <?php echo htmlspecialchars($error); ?>
          </div>
        <?php endif; ?>
      </form>
      
      <!-- Toggle between Login and Register -->
      <?php if ($mode === 'login'): ?>
      <div class="toggle-mode" style="margin-top: 25px; padding: 20px; background: rgba(46, 125, 50, 0.1); border-radius: 12px; border: 2px solid #2e7d32;">
        <p style="margin: 0 0 15px 0; color: #2e7d32; font-weight: 600; font-size: 1.1rem;">New User?</p>
        <a href="?mode=register" style="display: inline-block; padding: 15px 30px; background: #2e7d32; color: white; border-radius: 12px; text-decoration: none; font-weight: 600; font-size: 1.1rem; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);">
          <i class="fas fa-user-plus"></i> Register Now
        </a>
      </div>
      <?php else: ?>
      <div class="toggle-mode">
        <p>Already have an account? <a href="?mode=login" style="color: #2e7d32; font-weight: bold; text-decoration: underline;">Login here</a></p>
      </div>
      <?php endif; ?>
      
      <?php if ($mode === 'login'): ?>
      <div class="demo-credentials">
        <h4><i class="fas fa-info-circle"></i> Demo Credentials</h4>
        <p><strong>Username:</strong> admin</p>
        <p><strong>Password:</strong> 1234</p>
      </div>
      <?php endif; ?>
      
      <div class="footer-text">
        <i class="fas fa-leaf"></i> Pechay Leaf Detection System
      </div>
    </div>
  </div>
</body>
</html>
