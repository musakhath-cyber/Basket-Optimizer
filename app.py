import io
import os
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image
import pydantic
from pydantic import BaseModel, Field

app = FastAPI(title="Food Catalog Vision & Parser API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (reads GEMINI_API_KEY from environment)
client = genai.Client()


class CatalogItem(BaseModel):
    item_name: str = Field(description="Clean, normalized product name without promotional clutter.")
    category: str = Field(
        description="Assigned category: 'Produce', 'Proteins', 'Dairy', 'Pantry & Dry Goods', 'Beverages', or 'Packaging'."
    )
    pack_size: str = Field(description="Extracted weight, volume, or pack quantity, e.g., '10kg', '2L', '6x500g', 'Each'.")
    price: float = Field(description="Numerical price extracted from the catalog.")
    currency: str = Field(default="ZAR", description="Currency symbol or 3-letter code extracted.")
    confidence_score: float = Field(
        default=0.95,
        description="Confidence score between 0.0 and 1.0 based on text clarity.",
    )


class CatalogScanResponse(BaseModel):
    supplier_name: Optional[str] = Field(None, description="Detected vendor or supplier name.")
    currency: str = Field(default="ZAR")
    total_items_detected: int
    items: List[CatalogItem]


@app.post("/api/scan-catalog", response_model=CatalogScanResponse)
async def scan_catalog(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image or PDF.")

    contents = await file.read()

    try:
        # Load image with PIL to validate
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    prompt = """
    You are an expert commercial food procurement and catalog parsing system.
    Analyze this uploaded catalog / price sheet image:
    1. Extract all individual food and inventory items.
    2. Normalize product descriptions (remove excessive promo buzzwords).
    3. Accurately detect pack size (e.g., 5kg, 10x1L, 2.5kg, Tray of 30).
    4. Extract the exact item unit price.
    5. Categorize each item strictly into one of: Produce, Proteins, Dairy, Pantry & Dry Goods, Beverages, Packaging.
    6. Identify the supplier name and primary currency if visible.
    """

    try:
        # Request structured JSON matching the Pydantic schema
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CatalogScanResponse,
                temperature=0.1,
            ),
        )
        parsed_result = CatalogScanResponse.model_validate_json(response.text)
        return parsed_result

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Vision extraction failed: {str(err)}")


@app.get("/health")
def health_check():
    return {"status": "active"}
