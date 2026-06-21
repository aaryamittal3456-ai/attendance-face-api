from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import base64
import os
import json
import numpy as np
import psycopg2

app = Flask(__name__)

# ── PostgreSQL connection ──────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://attendance_db_tvyg_user:tiro6Kyyb6oFXphc6DJAjuWLithvDMZF@dpg-d8pce6pkh4rs7394g6hg-a.singapore-postgres.render.com/attendance_db_tvyg"
)

def get_db():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    return conn

# ── Create tables if not exist ─────────────────────────
def init_db():
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                roll_no VARCHAR(20) UNIQUE,
                email VARCHAR(100),
                section VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                student_id INT REFERENCES students(id),
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                class_name VARCHAR(100),
                subject VARCHAR(100),
                session_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                student_id INT REFERENCES students(id),
                session_id INT REFERENCES sessions(id),
                status VARCHAR(10) DEFAULT 'present',
                confidence FLOAT,
                marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        db.commit()
        cursor.close()
        db.close()
        print("Database tables ready.")
    except Exception as e:
        print(f"DB init error: {e}")


# ── MediaPipe face detector + embedder ─────────────────
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def get_face_crops(image_bgr):
    """Detect faces and return list of cropped face images (resized, grayscale-flattened vectors)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    crops = []
    if results.detections:
        h, w, _ = image_bgr.shape
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x = max(int(box.xmin * w), 0)
            y = max(int(box.ymin * h), 0)
            bw = int(box.width * w)
            bh = int(box.height * h)
            face_crop = image_bgr[y:y + bh, x:x + bw]
            if face_crop.size > 0:
                crops.append(face_crop)
    return crops


def get_embedding(face_crop):
    """Generate a simple but effective embedding using resized grayscale pixel histogram + HOG-like features."""
    face_resized = cv2.resize(face_crop, (100, 100))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Compute HOG features (lightweight, no heavy model needed)
    win_size = (100, 100)
    block_size = (20, 20)
    block_stride = (10, 10)
    cell_size = (10, 10)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(gray)
    return features.flatten()


# ── Health check ───────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "running", "message": "Attendance Face API is live"})


# ── Enroll a student ───────────────────────────────────
@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    name      = data.get("name")
    roll_no   = data.get("roll_no")
    email     = data.get("email", "")
    section   = data.get("section", "")
    image_b64 = data.get("image")

    if not all([name, roll_no, image_b64]):
        return jsonify({"success": False, "error": "name, roll_no and image are required"}), 400

    try:
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400

        crops = get_face_crops(image_bgr)
        if not crops:
            return jsonify({"success": False, "error": "No face detected in the image"}), 400

        embedding = get_embedding(crops[0]).tolist()

        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            INSERT INTO students (name, roll_no, email, section)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (roll_no) DO UPDATE
            SET name=EXCLUDED.name, email=EXCLUDED.email, section=EXCLUDED.section
        """, (name, roll_no, email, section))

        cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
        student_id = cursor.fetchone()[0]

        cursor.execute("DELETE FROM embeddings WHERE student_id = %s", (student_id,))
        cursor.execute("""
            INSERT INTO embeddings (student_id, embedding)
            VALUES (%s, %s)
        """, (student_id, json.dumps(embedding)))

        db.commit()
        cursor.close()
        db.close()

        return jsonify({"success": True, "message": f"{name} enrolled successfully"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Scan group photo ───────────────────────────────────
@app.route("/scan", methods=["POST"])
def scan():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "error": "image field is required"}), 400

    try:
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400

        crops = get_face_crops(image_bgr)
        if not crops:
            return jsonify({
                "success": True,
                "faces_detected": 0,
                "matched": [],
                "matched_count": 0
            })

        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            SELECT s.id, s.name, s.roll_no, s.email, e.embedding
            FROM students s
            JOIN embeddings e ON s.id = e.student_id
        """)
        students = cursor.fetchall()
        cursor.close()
        db.close()

        if not students:
            return jsonify({"success": False, "error": "No enrolled students found"}), 404

        known_embeddings = [np.array(json.loads(s[4])) for s in students]

        matched_students = []
        matched_ids = set()

        for crop in crops:
            face_embedding = get_embedding(crop)

            best_idx = None
            best_distance = float("inf")

            for idx, known in enumerate(known_embeddings):
                distance = np.linalg.norm(face_embedding - known)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            # Threshold tuned for HOG feature distance
            if best_idx is not None and best_distance < 15 and students[best_idx][0] not in matched_ids:
                student = students[best_idx]
                matched_ids.add(student[0])
                confidence = round(max(0, (1 - best_distance / 25) * 100), 2)
                matched_students.append({
                    "student_id": student[0],
                    "name": student[1],
                    "roll_no": student[2],
                    "email": student[3],
                    "confidence": confidence
                })

        return jsonify({
            "success": True,
            "faces_detected": len(crops),
            "matched": matched_students,
            "matched_count": len(matched_students)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Save attendance ────────────────────────────────────
@app.route("/save-attendance", methods=["POST"])
def save_attendance():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    class_name = data.get("class_name")
    subject    = data.get("subject")
    date       = data.get("date")
    students   = data.get("students", [])

    if not all([class_name, subject, date, students]):
        return jsonify({"success": False, "error": "class_name, subject, date and students are required"}), 400

    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            INSERT INTO sessions (class_name, subject, session_date)
            VALUES (%s, %s, %s) RETURNING id
        """, (class_name, subject, date))
        session_id = cursor.fetchone()[0]

        for student in students:
            cursor.execute("""
                INSERT INTO attendance (student_id, session_id, status, confidence)
                VALUES (%s, %s, 'present', %s)
            """, (student["student_id"], session_id, student.get("confidence", 0)))

        db.commit()
        cursor.close()
        db.close()

        return jsonify({
            "success": True,
            "session_id": session_id,
            "students_marked": len(students)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)