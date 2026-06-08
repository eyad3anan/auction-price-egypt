import sys
import os
import urllib.request
import zipfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ─────────────────────────────────────────────
# ONE-TIME AUTOMATIC VOLUME POPULATOR
# ─────────────────────────────────────────────
# If the volume is blank, pull down the zip directly inside the cloud host
if not os.path.exists(os.path.join(MODELS_DIR, "encoders.pkl")):
    print("Persistent volume detected as empty. Downloading model binaries...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # REPLACE THIS URL with your direct share/download URL for models.zip
    DOWNLOAD_URL = "https://your-storage-provider.com/s/direct-link-to-models.zip"
    zip_path = os.path.join(MODELS_DIR, "models.zip")
    
    try:
        # Download the zip file straight onto the Railway volume mount
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)
        
        # Unpack the contents right into the volume
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
            
        # Clean up the zip file to save disk space
        os.remove(zip_path)
        print("Model binaries successfully written to Railway Volume!")
    except Exception as download_error:
        print(f"Volume initialization failed: {str(download_error)}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

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
    condition:           str   = Field(..., example="Like New")
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
        return PredictionResponse(predicted_final_selling_price_egp=float(price))
    except Exception as e:
        error_msg = str(e)
        
        # Intercept numeric strings hidden inside the exception message
        clean_number = error_msg.replace("Pipeline Error:", "").strip()
        
        try:
            # If the error text is just the computed price, recover it safely
            fixed_price = float(clean_number)
            return PredictionResponse(predicted_final_selling_price_egp=fixed_price)
        except ValueError:
            # If it's a real script crash (and not just a returned number), show it
            raise HTTPException(status_code=500, detail=f"Actual Script Error: {error_msg}")

@app.get("/")
def root():
    return {
        "message": "Egyptian Auction Price Predictor API",
        "docs": "/docs",
        "predict": "/predict  [POST]",
        "health": "/health  [GET]"
    }