import pytest
from fastapi.testclient import TestClient
from app.main import app, SpotifyFeatures

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_valid_payload():
    """Test prediction endpoint with valid Spotify audio features."""
    payload = {
        "danceability": 0.7,
        "energy": 0.8,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.1,
        "instrumentalness": 0.0,
        "liveness": 0.2,
        "valence": 0.6,
        "tempo": 120.0,
        "duration_ms": 240000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "genre" in data
    assert "confidence" in data


def test_predict_endpoint_invalid_payload():
    """Test prediction endpoint with invalid payload (missing required fields)."""
    payload = {"danceability": 0.7}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
