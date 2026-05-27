import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi_simple_redis_cache.NaiveCache import NaiveCache
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


_default_model_path = (
    Path(__file__).resolve().parent.parent.parent
    / "distilbert-base-uncased-finetuned-sst2"
)
model_path = os.environ.get("MODEL_PATH", str(_default_model_path))

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    top_k=None,
)

logger = logging.getLogger(__name__)
LOCAL_REDIS_PORT = 6379
LOCAL_REDIS_URL = "redis://localhost"
REDIS_HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
REDIS_HOST_PORT = int(os.environ.get("REDIS_PORT", LOCAL_REDIS_PORT))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up Project API")
    yield
    logging.info("Shutting down Project API")


sub_application_sentiment_predict = FastAPI(lifespan=lifespan)
sub_application_sentiment_predict.add_middleware(
    NaiveCache,
    redis_host=REDIS_HOST_URL,
    redis_port=REDIS_HOST_PORT,
    redis_db=0,
    redis_prefix="fastapi-cache-project",
    excluded_paths=["/health"],
)

ML_THREAD_POOL = ThreadPoolExecutor(
    max_workers=4,  # Tune based on your CPU cores
    thread_name_prefix="ml-inference",
)


class SentimentRequest(BaseModel):
    text: list[str]


class Sentiment(BaseModel):
    label: str
    score: float


class SentimentResponse(BaseModel):
    # ... [Sentiment]
    predictions: list[list[Sentiment]]


@sub_application_sentiment_predict.post(
    "/bulk-predict", response_model=SentimentResponse
)
async def predict(sentiments: SentimentRequest):
    try:
        result = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                ML_THREAD_POOL,
                lambda: classifier(sentiments.text, truncation=True, max_length=512),
            ),
            timeout=10.0,
        )
        return {"predictions": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timeout")


@sub_application_sentiment_predict.get("/health")
async def health():
    return {"status": "healthy"}
