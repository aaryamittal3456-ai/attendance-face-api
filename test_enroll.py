import requests
import base64

with open('C:/Users/User/Pictures/Aarya.jpeg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post('http://127.0.0.1:5000/enroll', json={
    "name": "Aarya Mittal",
    "roll_no": "CS101",
    "image": image_b64
})

print(response.json())