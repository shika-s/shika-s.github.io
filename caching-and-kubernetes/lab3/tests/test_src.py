import pytest

from fastapi.testclient import TestClient
from src.main import app
from fastapi import status
from unittest.mock import patch
from datetime import datetime

from src.main import app

# ─── /lab/hello endpoint ────────────────────────────────────────────


class TestHelloEndpoint:

    def test_hello_valid_name(self):
        with TestClient(app) as client:
            response = client.get("/lab/hello", params={"name": "World"})
            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {"message": "Hello World"}

    def test_hello_missing_name(self):
        with TestClient(app) as client:
            response = client.get("/lab/hello")
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_hello_empty_name(self):
        with TestClient(app) as client:
            response = client.get("/lab/hello", params={"name": ""})
            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {"message": "Hello "}

    def test_hello_name_with_spaces(self):
        with TestClient(app) as client:
            response = client.get("/lab/hello", params={"name": "John Doe"})
            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {"message": "Hello John Doe"}

    def test_hello_post_not_allowed(self):
        with TestClient(app) as client:
            response = client.post("/lab/hello")
            assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


# ─── /lab/health endpoint ───────────────────────────────────────────


class TestHealthEndpoint:

    def test_health_returns_200(self):
        with TestClient(app) as client:
            response = client.get("/lab/health")
            assert response.status_code == status.HTTP_200_OK

    def test_health_returns_time_key(self):
        with TestClient(app) as client:
            response = client.get("/lab/health")
            assert "time" in response.json()

    def test_health_time_is_iso_format(self):
        with TestClient(app) as client:
            response = client.get("/lab/health")
            time_str = response.json()["time"]
            # ISO 8601 format should contain 'T' separator
            assert "T" in time_str

    def test_health_post_not_allowed(self):
        with TestClient(app) as client:
            response = client.post("/lab/health")
            assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_health(self):
        with TestClient(app) as client:
            mock_time_value = datetime(2020, 6,18,1,0,30)
            with patch("src.housing_predict.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_time_value
                response = client.get("/lab/health")
                assert response.status_code == status.HTTP_200_OK
                assert response.json() == {"time": mock_time_value.isoformat()}


def test_predict_basic():
    with TestClient(app) as lifespanned_client:
        data = {
            "MedInc": 1,
            "HouseAge": 1,
            "AveRooms": 3,
            "AveBedrms": 3,
            "Population": 3,
            "AveOccup": 5,
            "Latitude": 1,
            "Longitude": 1,
        }
        response = lifespanned_client.post(
            "/lab/predict",
            json=data,
        )
        assert response.status_code == 200
        assert isinstance(response.json()["prediction"], float)


def test_bulk_predict_basic():
    with TestClient(app) as lifespanned_client:
        data = {
            "houses": [
                {
                    "MedInc": 1,
                    "HouseAge": 1,
                    "AveRooms": 3,
                    "AveBedrms": 3,
                    "Population": 3,
                    "AveOccup": 5,
                    "Latitude": 1,
                    "Longitude": 1,
                },
                {
                    "MedInc": 1,
                    "HouseAge": 1,
                    "AveRooms": 3,
                    "AveBedrms": 3,
                    "Population": 3,
                    "AveOccup": 5,
                    "Latitude": 1,
                    "Longitude": 1,
                },
            ]
        }
        response = lifespanned_client.post(
            "/lab/bulk-predict",
            json=data,
        )
        assert response.status_code == 200
        assert isinstance(response.json()["predictions"], list)
        assert isinstance(response.json()["predictions"][0], float)
        assert isinstance(response.json()["predictions"][1], float)
def test_bulk_predict_returns_correct_count():
    with TestClient(app) as client:
        data = {
            "houses": [
                {"MedInc": 1, "HouseAge": 1, "AveRooms": 3, "AveBedrms": 3, "Population": 3, "AveOccup": 5, "Latitude": 1, "Longitude": 1},
                {"MedInc": 5, "HouseAge": 10, "AveRooms": 6, "AveBedrms": 2, "Population": 500, "AveOccup": 3, "Latitude": 34, "Longitude": -118},
                {"MedInc": 8, "HouseAge": 30, "AveRooms": 7, "AveBedrms": 1, "Population": 300, "AveOccup": 2, "Latitude": 37, "Longitude": -122},
            ]
        }
        response = client.post("/lab/bulk-predict", json=data)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 3


def test_bulk_predict_missing_houses_key():
    with TestClient(app) as client:
        response = client.post("/lab/bulk-predict", json={"data": [{}]})
        assert response.status_code == 422


def test_bulk_predict_missing_field_in_house():
    with TestClient(app) as client:
        data = {"houses": [{"MedInc": 1, "HouseAge": 1}]}
        response = client.post("/lab/bulk-predict", json=data)
        assert response.status_code == 422


def test_bulk_predict_no_body():
    with TestClient(app) as client:
        response = client.post("/lab/bulk-predict")
        assert response.status_code == 422


def test_bulk_predict_wrong_method():
    with TestClient(app) as client:
        response = client.get("/lab/bulk-predict")
        assert response.status_code == 405


# Cache tests
def test_predict_cache_miss_then_hit():
    with TestClient(app) as client:
        data = {
            "MedInc": 8.3,
            "HouseAge": 41.0,
            "AveRooms": 6.98,
            "AveBedrms": 1.02,
            "Population": 322.0,
            "AveOccup": 2.55,
            "Latitude": 37.88,
            "Longitude": -122.23,
        }

        # First request — cache miss
        response1 = client.post("/lab/predict", json=data)
        assert response1.status_code == 200
        assert response1.headers["x-cache-hit"] == "False"

        # Second identical request — cache hit
        response2 = client.post("/lab/predict", json=data)
        assert response2.status_code == 200
        assert response2.headers["x-cache-hit"] == "True"

        # Cached response should be faster
        assert float(response2.headers["x-processing-time"]) < float(response1.headers["x-processing-time"])

        # Both should return the same prediction
        assert response1.json() == response2.json()


def test_bulk_predict_cache_miss_then_hit():
    with TestClient(app) as client:
        data = {
            "houses": [
                {"MedInc": 5, "HouseAge": 20, "AveRooms": 6, "AveBedrms": 1, "Population": 500, "AveOccup": 3, "Latitude": 35, "Longitude": -120},
                {"MedInc": 3, "HouseAge": 15, "AveRooms": 5, "AveBedrms": 1, "Population": 800, "AveOccup": 4, "Latitude": 36, "Longitude": -119},
            ]
        }

        response1 = client.post("/lab/bulk-predict", json=data)
        assert response1.headers["x-cache-hit"] == "False"

        response2 = client.post("/lab/bulk-predict", json=data)
        assert response2.headers["x-cache-hit"] == "True"

        assert response1.json() == response2.json()


def test_different_input_is_cache_miss():
    with TestClient(app) as client:
        data1 = {"MedInc": 5, "HouseAge": 20, "AveRooms": 6, "AveBedrms": 1, "Population": 500, "AveOccup": 3, "Latitude": 35, "Longitude": -120}
        data2 = {"MedInc": 9, "HouseAge": 20, "AveRooms": 6, "AveBedrms": 1, "Population": 500, "AveOccup": 3, "Latitude": 35, "Longitude": -120}

        client.post("/lab/predict", json=data1)

        # Different input should be a miss even after caching data1
        response = client.post("/lab/predict", json=data2)
        assert response.headers["x-cache-hit"] == "False"


def test_cache_bypass_with_no_store():
    with TestClient(app) as client:
        data = {"MedInc": 7, "HouseAge": 30, "AveRooms": 6, "AveBedrms": 1, "Population": 400, "AveOccup": 2, "Latitude": 34, "Longitude": -118}

        # Cache it first
        client.post("/lab/predict", json=data)

        # Send with no-store header — should bypass cache
        response = client.post("/lab/predict", json=data, headers={"cache-control": "no-store"})
        assert response.headers.get("x-cache-hit") is None or response.headers["x-cache-hit"] == "False"




