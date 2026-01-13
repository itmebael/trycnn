from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import random
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change-this-secret"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def simulate_detection(image_path: str) -> dict:
    conditions = ["Healthy", "Diseased"]
    return {
        "condition": random.choice(conditions),
        "confidence": random.randint(75, 95),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
    }

def require_login():
    if not session.get("user"):
        return redirect(url_for("login"))
    return None

@app.route("/login", methods=["GET", "POST", "HEAD"]) 
@app.route("/", methods=["GET", "POST", "HEAD"]) 
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid username or password!"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"]) 
def dashboard():
    gate = require_login()
    if gate is not None:
        return gate

    page = request.args.get("page", "dashboard")
    upload_status = ""
    detection_result = None

    if request.method == "POST" and page == "upload":
        file = request.files.get("leafImage")
        if not file or file.filename == "":
            upload_status = "File is required."
        elif not allowed_file(file.filename):
            upload_status = "File is not an image."
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            upload_status = "File uploaded successfully."
            detection_result = simulate_detection(save_path)

    # Build results from uploads folder
    results = []
    if page == "results" and os.path.isdir(UPLOAD_FOLDER):
        for name in sorted(os.listdir(UPLOAD_FOLDER)):
            lower = name.lower()
            if any(lower.endswith("." + ext) for ext in ALLOWED_EXTENSIONS):
                fpath = os.path.join(UPLOAD_FOLDER, name)
                results.append({
                    "filename": name,
                    "path": os.path.join("uploads", name).replace("\\", "/"),
                    "timestamp": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S"),
                    "condition": random.choice(["Healthy", "Diseased"]),
                    "confidence": random.randint(75, 95),
                })

    return render_template(
        "dashboard.html",
        page=page,
        upload_status=upload_status,
        detection_result=detection_result,
        results=results,
    )

@app.get("/delete")
def delete_file():
    gate = require_login()
    if gate is not None:
        return gate
    filename = request.args.get("file", "")
    safe_name = os.path.basename(filename)
    fpath = os.path.join(UPLOAD_FOLDER, safe_name)
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
            flash("File deleted successfully.")
        except OSError:
            flash("Error deleting file.")
    else:
        flash("File not found.")
    return redirect(url_for("dashboard", page="results"))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    print("Running Flask app with simulated detection (no PyTorch required)")
    app.run(debug=True)
