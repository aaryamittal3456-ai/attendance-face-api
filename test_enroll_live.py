import requests
import base64

# Use any clear photo of a face
with open('C:/Users/User/Pictures/Aarya.jpeg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    'https://attendance-face-api-9p4f.onrender.com/enroll',
    json={
        "name": "Aarya Mittal",
        "roll_no": "CS101",
        "image": image_b64
    }
)

print("Status code:", response.status_code)
print(response.json())