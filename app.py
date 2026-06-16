from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
import os
import json
import numpy as np
import mysql.connector

app = Flask(__name__)

# ── MySQL connection ───────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "MyNewPasswordUpdated",  # ← change this
    "database": "attendance_db"
}


def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ── Enroll a student ───────────────────────────────────
@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.json
    name      = data["name"]
    roll_no   = data["roll_no"]
    email     = data.get("email", "")
    section   = data.get("section", "")
    image_b64 = data["image"]

    # Save temp image
    img_bytes = base64.b64decode(image_b64)
    temp_path = f"temp_{roll_no}.jpg"
    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    try:
        # Get face embedding
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True
        )[0]["embedding"]

        db = get_db()
        cursor = db.cursor()

        # Insert student
        cursor.execute("""
            INSERT INTO students (name, roll_no, email, section)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            name=VALUES(name), email=VALUES(email), section=VALUES(section)
        """, (name, roll_no, email, section))

        # Get student id
        cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
        student_id = cursor.fetchone()[0]

        # Save embedding
        cursor.execute("""
            INSERT INTO embeddings (student_id, embedding)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE embedding=VALUES(embedding)
        """, (student_id, json.dumps(embedding)))

        db.commit()
        cursor.close()
        db.close()

        return jsonify({"success": True, "message": f"{name} enrolled successfully"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Scan group photo ───────────────────────────────────
@app.route("/scan", methods=["POST"])
def scan():
    data = request.json
    image_b64 = data["image"]

    img_bytes = base64.b64decode(image_b64)
    temp_path = "temp_group.jpg"
    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    try:
        # Detect all faces in group photo
        faces = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=False
        )

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

        matched_students = []

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

            if best_score < 10 and best_match:
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
        return jsonify({"success": False, "error": str(e)})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Save attendance record ─────────────────────────────
@app.route("/save-attendance", methods=["POST"])
def save_attendance():
    data = request.json
    class_name = data["class_name"]
    subject    = data["subject"]
    date       = data["date"]
    students   = data["students"]  # list from /scan response

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
        """, (student["student_id"], session_id, student["confidence"]))

    db.commit()
    cursor.close()
    db.close()

    return jsonify({
        "success": True,
        "session_id": session_id,
        "students_marked": len(students)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)