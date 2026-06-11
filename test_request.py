import requests

payload = {
    "danceability": 0.7,
    "energy": 0.8,
    "key": 5,
    "loudness": -5.2,
    "mode": 1,
    "speechiness": 0.05,
    "acousticness": 0.2,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "valence": 0.6,
    "tempo": 120.0,
    "duration_ms": 210000
}

r = requests.post("http://127.0.0.1:5001/predict", json=payload)
print("Status code:", r.status_code)
print("Response:", r.json())

