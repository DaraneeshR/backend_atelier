"""
backend/server.py
─────────────────────────────────────────────────────
Flask API server for Vision-Based Smart Tailoring System.
Replaces Streamlit session-state logic with JWT-based REST endpoints.
"""

import sys, os

# Allow imports from the project root (pose_estimator, database, etc.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from functools import wraps
from datetime import datetime, timedelta, timezone

import jwt
from flask import Flask, request, jsonify
from flask_cors import CORS

from database import Database
from auth.auth_manager import hash_password, verify_password, create_user, authenticate
from measurement_calculator import MeasurementCalculator
from adjustment_engine import AdjustmentEngine
from size_classifier import SizeClassifier
from gender_config import load_config

# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        return response, 200
app.config["SECRET_KEY"] = os.environ.get("JWT_SECRET", "dev-secret-change-in-production")
cors_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

CORS(
    app,
    resources={r"/api/*": {"origins": cors_origins}},
    supports_credentials=True
)

db = Database()
db.init_db()

calc = MeasurementCalculator()
engine = AdjustmentEngine()
classifier = SizeClassifier()

@app.route("/health")
def health():
    return {"status": "running"}

@app.before_request
def log_request():
    print("Incoming:", request.method, request.path)

# ── JWT helpers ──────────────────────────────────────────────────────────────

def create_token(user: dict) -> str:
    payload = {
        "sub": str(user["id"]),       # PyJWT v2.11+ requires sub to be a string
        "name": user["name"],
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=24),
    }
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    # PyJWT v1 returns bytes, v2 returns str — ensure str
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        if not token:
            return jsonify({"error": "Token required"}), 401
        try:
            payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            # Convert sub back to int for DB lookups
            payload["sub"] = int(payload["sub"])
            request.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError as e:
            print(f"[JWT] Invalid token error: {e}")
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Auth endpoints ───────────────────────────────────────────────────────────

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    role = data.get("role", "user")

    errors = []
    if not name:
        errors.append("Name is required.")
    if not email or "@" not in email:
        errors.append("Valid email is required.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if role not in ("user", "tailor"):
        errors.append("Role must be 'user' or 'tailor'.")
    if errors:
        return jsonify({"errors": errors}), 400

    conn = db.get_connection()
    try:
        create_user(conn, name, email, password, role)
        return jsonify({"message": "Account created successfully."}), 201
    except Exception as e:
        if "UNIQUE" in str(e):
            return jsonify({"errors": ["An account with this email already exists."]}), 409
        return jsonify({"errors": [str(e)]}), 500
    finally:
        conn.close()

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password required."}), 400

    conn = db.get_connection()
    try:
        user = authenticate(conn, email, password)
        if not user:
            return jsonify({"error": "Incorrect email or password."}), 401
        token = create_token(user)
        return jsonify({
            "token": token,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
            },
        })
    finally:
        conn.close()


@app.route("/api/auth/me", methods=["GET"])
@token_required
def me():
    return jsonify({
        "id": request.user["sub"],
        "name": request.user["name"],
        "role": request.user["role"],
    })


# ── Measurement endpoints ────────────────────────────────────────────────────

@app.route("/api/measurements", methods=["GET"])
@token_required
def get_measurements():
    user_id = request.user["sub"]
    records = db.get_measurements_by_user(user_id)
    return jsonify(records)


@app.route("/api/measurements/<int:record_id>", methods=["GET"])
@token_required
def get_measurement(record_id):
    record = db.get_record_by_id(record_id)
    if not record:
        return jsonify({"error": "Not found"}), 404
    return jsonify(record)


@app.route("/api/measurements/<int:record_id>", methods=["DELETE"])
@token_required
def delete_measurement(record_id):
    db.delete_record(record_id)
    return jsonify({"message": "Deleted"}), 200


@app.route("/api/measurements", methods=["POST"])
@token_required
def save_measurement():
    """Save a completed measurement (body + garment + meta) to the database."""
    data = request.get_json()
    user_id = request.user["sub"]

    record_id = db.save_measurement(
        name=data.get("name", ""),
        height_cm=data.get("height_cm", 170),
        gender=data.get("gender", "Male"),
        fit_preference=data.get("fit_preference", "Regular"),
        sleeve_type=data.get("sleeve_type", "Full Sleeve"),
        body_measurements=data.get("body_measurements", {}),
        garment_measurements=data.get("garment_measurements", {}),
        recommended_size=data.get("recommended_size", "M"),
        confidence_score=data.get("confidence_score", 0),
        user_id=user_id,
    )
    return jsonify({"id": record_id, "message": "Saved"}), 201


# ── Compute endpoint (landmarks → measurements) ─────────────────────────────

@app.route("/api/compute", methods=["POST"])
@token_required
def compute_measurements():
    """
    Accepts averaged pose landmarks + user params, returns body + garment measurements.

    Expected JSON body:
        landmarks: dict of { "0": {x, y, z, v}, "1": {x, y, z, v}, ... }
                   (33 MediaPipe landmarks, normalized 0-1, averaged across frames)
                   Keys are string indices "0"-"32".
        confidence: float (0-1) — multi-frame confidence score
        frame_width: int
        frame_height: int
        height_cm: float
        gender: str
        weight_kg: float | null
        fit_preference: str
        sleeve_type: str
    """
    data = request.get_json()
    landmarks_raw = data.get("landmarks", {})
    confidence = data.get("confidence", 0.0)
    frame_w = data.get("frame_width", 1280)
    frame_h = data.get("frame_height", 720)
    height_cm = data.get("height_cm", 170)
    gender = data.get("gender", "Male")
    weight_kg = data.get("weight_kg")
    fit_pref = data.get("fit_preference", "Regular")
    sleeve_type = data.get("sleeve_type", "Full Sleeve")

    # Convert string-keyed dict to int-keyed dict matching MeasurementCalculator format
    # MeasurementCalculator.compute() expects: { int_index: {'x':…,'y':…,'z':…,'v':…} }
    averaged_lm = {}
    for key, val in landmarks_raw.items():
        averaged_lm[int(key)] = {
            "x": val.get("x", 0),
            "y": val.get("y", 0),
            "z": val.get("z", 0),
            "v": val.get("v", val.get("visibility", 0)),
        }

    gender_cfg = load_config(gender)
    gender_cfg["_gender"] = gender.lower()

    body_meas = calc.compute(averaged_lm, frame_w, frame_h, height_cm, gender_cfg,
                             weight_kg=weight_kg)
    garment_meas = engine.compute(body_meas, gender_cfg, fit_pref, sleeve_type)
    size_label, size_range = classifier.classify(
        garment_meas.get("garment_chest", 0), gender
    )

    return jsonify({
        "body_measurements": body_meas,
        "garment_measurements": garment_meas,
        "recommended_size": size_label,
        "size_range": size_range,
        "confidence_score": confidence,
    })


# ── Analytics endpoint ───────────────────────────────────────────────────────

@app.route("/api/analytics", methods=["GET"])
@token_required
def get_analytics():
    user_id = request.user["sub"]
    analytics = db.get_analytics(user_id)
    return jsonify(analytics)


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Flask API server on port {port}")
    app.run(host="0.0.0.0", debug=True, port=port)
