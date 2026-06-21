from flask import Flask, request, jsonify
import face_recognition
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
        img_bytes = base64.b64decode(image_b64)
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # Load image and get face encoding (dlib-based, lightweight)
        image = face_recognition.load_image_file(temp_path)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            return jsonify({"success": False, "error": "No face detected in the image"}), 400

        embedding = face_encodings[0].tolist()

        db = get_db()
        cursor = db.cursor()

        # Insert or update student
        cursor.execute("""
            INSERT INTO students (name, roll_no, email, section)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (roll_no) DO UPDATE
            SET name=EXCLUDED.name, email=EXCLUDED.email, section=EXCLUDED.section
        """, (name, roll_no, email, section))

        cursor.execute("SELECT id FROM students WHERE roll_no = %s", (roll_no,))
        student_id = cursor.fetchone()[0]

        # Delete old embedding and insert new
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
        img_bytes = base64.b64decode(image_b64)
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
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

        known_encodings = [np.array(json.loads(s[4])) for s in students]

        matched_students = []
        matched_ids = set()

        for face_encoding in face_encodings:
            # Compare against all known faces
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(np.argmin(distances))
            best_distance = distances[best_idx]

            # Lower distance = better match. 0.6 is the standard threshold.
            if best_distance < 0.6:
                student = students[best_idx]
                if student[0] not in matched_ids:
                    matched_ids.add(student[0])
                    confidence = round((1 - best_distance) * 100, 2)
                    matched_students.append({
                        "student_id": student[0],
                        "name": student[1],
                        "roll_no": student[2],
                        "email": student[3],
                        "confidence": confidence
                    })

        return jsonify({
            "success": True,
            "faces_detected": len(face_encodings),
            "matched": matched_students,
            "matched_count": len(matched_students)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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