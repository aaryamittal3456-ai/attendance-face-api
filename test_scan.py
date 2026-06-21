import requests
import base64

# Use any photo that has your face in it
with open('C:/Users/User/Pictures/Aarya.jpeg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post('http://127.0.0.1:5000/scan', json={
    "image": image_b64
})

print(response.json())