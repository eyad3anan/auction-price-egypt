"""
app/main.py — FastAPI prediction service for Egyptian Auction Price Prediction.

Run locally:
    uvicorn app.main:app --reload --port 8000

POST /predict  →  accepts listing features as JSON, returns predicted final_selling_price
GET  /health   →  liveness check
GET  /docs     →  interactive Swagger UI
"""

import sys
import os

# Clean, safe absolute system path insertion for production containers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

# This will now find your scripts directory without throwing errors
from scripts.predict import predict_single

app = FastAPI(
    title="Egyptian Auction Price Predictor",
    description="Predicts the final selling price (EGP) for items listed on Egyptian online auctions.",
    version="1.0.0",
)


# ─────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────
class AuctionListing(BaseModel):
    category:            str   = Field(..., example="Mobile Accessories")
    subcategory:         str   = Field(..., example="Cases & Protection")
    brand:               str   = Field(..., example="Apple")
    condition:           str   = Field(..., example="Like New",
                                      description="One of: For Parts, Poor, Fair, Good, Very Good, Excellent, Like New, New")
    product_age:         int   = Field(..., ge=0, example=6)
    starting_price:      float = Field(..., gt=0, example=500.0)
    auction_duration:    int   = Field(..., ge=1, le=30, example=7)
    listing_day_of_week: str   = Field(..., example="Saturday")
    listing_hour:        int   = Field(..., ge=0, le=23, example=20)
    seller_rating:       float = Field(..., ge=0, le=5, example=4.5)
    seller_total_sales:  int   = Field(..., ge=0, example=50)
    seller_account_age:  int   = Field(..., ge=0, example=24)
    verified_seller:     int   = Field(..., ge=0, le=1, example=1)


class PredictionResponse(BaseModel):
    predicted_final_selling_price_egp: float
    currency: str = "EGP"
    model_version: str = "1.0.0"


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Egyptian Auction Price Predictor"}


@app.post("/predict", response_model=PredictionResponse)
def predict(listing: AuctionListing):
    try:
        input_dict = listing.model_dump()
        price = predict_single(input_dict)
        return PredictionResponse(predicted_final_selling_price_egp=price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Egyptian Auction Price Predictor API",
        "docs": "/docs",
        "predict": "/predict  [POST]",
        "health": "/health  [GET]"
    }