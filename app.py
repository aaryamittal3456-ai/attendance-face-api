from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
import os
import json
import numpy as np
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# ── MySQL connection ───────────────────────────────────
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "MyNewPasswordUpdated"),
    "database": os.environ.get("DB_NAME", "attendance_db"),
    "port": int(os.environ.get("DB_PORT", 3306))
}


def get_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        raise Exception(f"Database connection failed: {str(e)}")


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

    temp_path = f"temp_{roll_no}.jpg"

    try:
        # Decode and save temp image
        img_bytes = base64.b64decode(image_b64)
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # Get face embedding
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True
        )
        embedding = result[0]["embedding"]

        db = get_db()
        cursor = db.cursor()

        # Insert or update student
        cursor.execute("""
            INSERT INTO students (name, roll_no, email, section)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            name=VALUES(name), email=VALUES(email), section=VALUES(section)
        """, (name, roll_no, email, section))

        # Get student id
        cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"success": False, "error": "Failed to retrieve student ID"}), 500
        student_id = row[0]

        # Save embedding (delete old first to avoid duplicates)
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

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Scan group photo ───────────────────────────────────
@app.route("/scan", methods=["POST"])
def scan():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "error": "image field is required"}), 400

    temp_path = "temp_group.jpg"

    try:
        # Decode and save temp image
        img_bytes = base64.b64decode(image_b64)
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # Detect all faces in group photo
        faces = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=False
        )

        if not faces:
            return jsonify({
                "success": True,
                "faces_detected": 0,
                "matched": [],
                "matched_count": 0,
                "message": "No faces detected in the image"
            })

        # Load all embeddings from MySQL
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
            return jsonify({
                "success": False,
                "error": "No enrolled students found in database"
            }), 404

        matched_students = []
        matched_ids = set()  # avoid duplicate matches

        for face in faces:
            face_embedding = np.array(face["embedding"])
            best_match = None
            best_score = float("inf")

            for student in students:
                stored = np.array(json.loads(student[4]))
                distance = np.linalg.norm(face_embedding - stored)
                if distance < best_score:
                    best_score = distance
                    best_match = student

            if best_score < 10 and best_match and best_match[0] not in matched_ids:
                matched_ids.add(best_match[0])
                matched_students.append({
                    "student_id": best_match[0],
                    "name": best_match[1],
                    "roll_no": best_match[2],
                    "email": best_match[3],
                    "confidence": round((1 - best_score / 20) * 100, 2)
                })

        return jsonify({
            "success": True,
            "faces_detected": len(faces),
            "matched": matched_students,
            "matched_count": len(matched_students)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Save attendance record ─────────────────────────────
@app.route("/save-attendance", methods=["POST"])
def save_attendance():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    class_name = data.get("class_name")
    subject    = data.get("subject")
    date       = data.get("date")
    students   = data.get("students", [])

    if not all([class_name, subject, date]):
        return jsonify({"success": False, "error": "class_name, subject and date are required"}), 400

    if not students:
        return jsonify({"success": False, "error": "No students to mark attendance for"}), 400

    try:
        db = get_db()
        cursor = db.cursor()

        # Create session
        cursor.execute("""
            INSERT INTO sessions (class_name, subject, session_date)
            VALUES (%s, %s, %s)
        """, (class_name, subject, date))
        session_id = cursor.lastrowid

        # Insert attendance for each matched student
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)