# Pechay Detection System (Flask)

A Flask port of your PHP app with login, dashboard, upload/detect (simulated), results listing, and delete.

## Requirements

- Python 3.10+
- pip

## Setup

```
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Run

```
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Auth

- Username: `admin`
- Password: `1234`

## Project structure

- `app.py`: Flask app and routes
- `templates/login.html`: Login UI
- `templates/dashboard.html`: Dashboard + upload + results
- `uploads/`: Uploaded images

## Notes

- Detection is simulated in `simulate_detection()`; plug in your model when ready.
- Static files are inlined for simplicity; you can extract CSS/JS later.
