"""
STPA - Flask Dashboard Application
=====================================
Interactive web dashboard for visualizing STPA simulation results.

Run:
    python dashboard/app.py

Serves:
    / — Main dashboard page
    /api/simulate — Proxy to FastAPI simulation
"""

from flask import Flask, render_template, jsonify, request
import requests
import json
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), ".."))

STPA_API = "http://localhost:8000"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json()
    try:
        resp = requests.post(f"{STPA_API}/simulate", json=data, timeout=60)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stress-test", methods=["POST"])
def stress_test():
    data = request.get_json()
    try:
        resp = requests.post(f"{STPA_API}/stress-test", json=data, timeout=60)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scenarios")
def scenarios():
    try:
        resp = requests.get(f"{STPA_API}/scenarios", timeout=10)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
