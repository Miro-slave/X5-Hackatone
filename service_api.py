
from fastapi import FastAPI, Request, HTTPException
import pandas as pd
from pydantic import BaseModel
import uvicorn
from model import Predictor
from dataclasses import dataclass


predictor = Predictor()
app = FastAPI()


class PredictionInput(BaseModel):
    input: str


@app.on_event("startup")
async def startup_event():
    await predictor.start()


@app.post("/api/predict")
async def predict_model(input_data: PredictionInput):
    if not input_data.input.strip():
        return []
    response = await predictor.get_request(input_data.input)
    return response 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    