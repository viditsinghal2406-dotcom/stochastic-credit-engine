"""
STPA - Run Script
==================
Start both the FastAPI backend and Flask dashboard with a single command.

Usage:
    python run.py

Then visit:
    Dashboard:  http://localhost:5000
    API Docs:   http://localhost:8000/docs
    GitHub Page: open index.html in browser
"""

import subprocess
import sys
import os
import time
import threading

def run_api():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def run_dashboard():
    time.sleep(2)  # Wait for API to start
    subprocess.run([sys.executable, "dashboard/app.py"])

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║          STPA — Stochastic Path Analyzer v2          ║
║          NovaCred Bank Credit Risk Engine            ║
╠══════════════════════════════════════════════════════╣
║  API:       http://localhost:8000                    ║
║  API Docs:  http://localhost:8000/docs               ║
║  Dashboard: http://localhost:5000                    ║
║  GitHub:    open index.html in browser               ║
╚══════════════════════════════════════════════════════╝
    """)

    t1 = threading.Thread(target=run_api, daemon=True)
    t2 = threading.Thread(target=run_dashboard, daemon=True)
    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down STPA...")
