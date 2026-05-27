curl -X GET "http://127.0.0.1:8000/lab/health" 

curl -X GET "http://127.0.0.1:8000/lab/hello?name=World"

curl -X POST "http://127.0.0.1:8000/lab/predict" \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 1, "HouseAge": 1, "AveRooms": 3, "AveBedrms": 3, "Population": 3, "AveOccup": 5, "Latitude": 1, "Longitude": 1}'

curl -X POST "http://127.0.0.1:8000/lab/predict" \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 0, "HouseAge": 1, "AveRooms": 3, "AveBedrms": 3, "Population": 3, "AveOccup": 5, "Latitude": 1, "Longitude": 1}'


curl -X GET "http://127.0.0.1:8000/lab/docs"

curl -X POST "http://127.0.0.1:8000/lab/bulk-predict" \
-H "Content-Type: application/json" \
-d '{
  "houses": [
    {
      "MedInc": 1,
      "HouseAge": 1,
      "AveRooms": 3,
      "AveBedrms": 3,
      "Population": 3,
      "AveOccup": 5,
      "Latitude": 1,
      "Longitude": 1
    },
    {
      "MedInc": 0,
      "HouseAge": 1,
      "AveRooms": 3,
      "AveBedrms": 3,
      "Population": 3,
      "AveOccup": 5,
      "Latitude": 1,
      "Longitude": 1
    }
  ]
}'

curl  -X POST http://localhost:8000/lab/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}'
