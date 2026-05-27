from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from src.housing_predict import lifespan_mechanism, sub_application_housing_predict

@asynccontextmanager
async def main_lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        # Manage the lifecycle of sub_app
        await stack.enter_async_context(
            lifespan_mechanism(sub_application_housing_predict)
        )
        yield


app = FastAPI(lifespan=main_lifespan)


app.mount("/lab", sub_application_housing_predict)



# /docs endpoint is defined by FastAPI automatically
# /openapi.json returns a json object automatically by FastAPI
