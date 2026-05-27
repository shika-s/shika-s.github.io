import logging
import pandas as pd
import random
import os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from joblib import load
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List
from fastapi_simple_redis_cache.NaiveCache import NaiveCache

logger = logging.getLogger(__name__)
model = None

LOCAL_REDIS_URL = "redis://127.0.0.1"
LOCAL_REDIS_PORT = 6379

@asynccontextmanager
async def lifespan_mechanism(app: FastAPI):
    logging.info("Starting up Lab3 API")

    # Load the Model on Startup
    global model
    model = load("model_pipeline.pkl")

    yield
    # We don't need a shutdown event for our system, but we could put something
    # here after the yield to deal with things during shutdown
    logging.info("Shutting down Lab3 API")


sub_application_housing_predict = FastAPI(lifespan=lifespan_mechanism)

REDIS_HOST_URL = os.getenv("REDIS_URL",LOCAL_REDIS_URL)  # TODO: replace this according to the Lab Requirements
REDIS_HOST_PORT = os.getenv("REDIS_PORT", LOCAL_REDIS_PORT)  # TODO: replace this according to the Lab Requirements
KEY_PREFIX = "w255-cache-prediction"

sub_application_housing_predict.add_middleware(
    NaiveCache,
    redis_host=REDIS_HOST_URL,
    redis_port=REDIS_HOST_PORT,
    redis_db=0,
    redis_prefix=KEY_PREFIX,
    excluded_paths=["/lab/health"] # check if needed
)  


class House(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"
    )
    @classmethod
    def should_be_greater_than_zero(cls, value: float, info) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name}  should be greater than zero")
        return value

    @field_validator("Latitude")
    @classmethod
    def latitude_must_be_valid(cls, value: float) -> float:
        if not (-90 <= value <= 90):
            raise ValueError("Invalid value for Latitude")
        return value

    @field_validator("Longitude")
    @classmethod
    def longitude_must_be_valid(cls, value: float) -> float:
        if not (-180 <= value <= 180):
            raise ValueError("Invalid value for Longitude")
        return value


class ListHouses(BaseModel):
    houses: List[House] = Field(min_length=1) # reject empty lists


# Response Model Validations
class SingleHousePrediction(BaseModel):
    prediction: float = Field(ge=0) # house price cannot be negative

class MultiHousePrediction(BaseModel):
    predictions: List[float] = Field(min_length=1)

    @field_validator("predictions")
    @classmethod
    def all_non_negative(cls, v):
        if any(p < 0 for p in v):
            raise ValueError("Predictions cannot be negative")
        return v



# Do not change this function name.
# See the Input Vectorization subsection in the readme for more instructions


@sub_application_housing_predict.get("/health")
def health():
    return {"time": datetime.now().isoformat()}

@sub_application_housing_predict.get("/hello")
def hello(name: str = Query(...)):
    return {"message": f"Hello {name}"}



@sub_application_housing_predict.post("/predict", response_model=SingleHousePrediction)
async def predict(house: House):
    df = pd.DataFrame([house.model_dump()])
    prediction = float(model.predict(df)[0])
    return SingleHousePrediction(prediction=prediction)


@sub_application_housing_predict.post("/bulk-predict", response_model=MultiHousePrediction)
async def multi_predict(request: ListHouses):
    data = list(map(lambda house: house.model_dump(), request.houses))    
    # data = [house.model_dump() for house in request.houses]
    predictions = model.predict(pd.DataFrame(data))
    return MultiHousePrediction(predictions=predictions.tolist())

    pass
