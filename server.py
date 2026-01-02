#!/usr/bin/env python3
"""
PaddleOCR HTTP Server with full OCR functionality
Provides complete OCR with text extraction
"""

import base64
import io
import os
from logging import getLogger
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from paddleocr import PaddleOCR, PPStructureV3
import uvicorn
import numpy as np
from PIL import Image

logger = getLogger(__name__)

# Configuration from environment variables
HOST = os.environ.get("OCR_HOST", "0.0.0.0")
PORT = int(os.environ.get("OCR_PORT", "8080"))
LOG_LEVEL = os.environ.get("OCR_LOG_LEVEL", "info")
LANG = os.environ.get("OCR_LANG", "en")

app = FastAPI(
    title="PaddleOCR Server",
    version="1.0.0",
    description="OCR server with text extraction and layout analysis using PaddleOCR v3+",
)


# =============================================================================
# Request Models
# =============================================================================

class OCRRequest(BaseModel):
    """Request model for OCR endpoints"""
    file: str = Field(..., description="Base64-encoded image data or HTTP(S) URL to image")
    fileType: int = Field(1, description="File type: 0 for PDF, 1 for image")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"file": "https://example.com/image.png", "fileType": 1},
                {"file": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "fileType": 1},
            ]
        }
    }


# =============================================================================
# Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Server status", examples=["healthy"])
    service: str = Field(..., description="Service name", examples=["PaddleOCR Server"])


class OCRPageResult(BaseModel):
    """OCR result for a single page/image"""
    input_path: str | None = Field(None, description="Input file path (null for in-memory images)")
    page_index: int | None = Field(None, description="Page index for multi-page documents")
    rec_texts: list[str] = Field(default_factory=list, description="List of recognized text strings")
    rec_scores: list[float] = Field(default_factory=list, description="Confidence scores for each text region")
    rec_polys: list[list[list[int]]] = Field(
        default_factory=list,
        description="Bounding polygons as list of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )
    rec_boxes: list[list[int]] = Field(
        default_factory=list,
        description="Bounding boxes as [x1, y1, x2, y2]"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_path": None,
                    "page_index": 0,
                    "rec_texts": ["INVOICE", "Date: 2024-01-15", "Total: $150.00"],
                    "rec_scores": [0.98, 0.95, 0.97],
                    "rec_polys": [
                        [[100, 50], [200, 50], [200, 80], [100, 80]],
                        [[100, 100], [250, 100], [250, 130], [100, 130]],
                        [[100, 150], [230, 150], [230, 180], [100, 180]],
                    ],
                    "rec_boxes": [
                        [100, 50, 200, 80],
                        [100, 100, 250, 130],
                        [100, 150, 230, 180],
                    ],
                }
            ]
        }
    }


class LayoutDetectionResult(BaseModel):
    """Layout detection result with regions"""
    boxes: list[list[float]] = Field(default_factory=list, description="Bounding boxes [x1, y1, x2, y2]")
    labels: list[str] = Field(default_factory=list, description="Region labels (text, table, figure, etc.)")
    scores: list[float] = Field(default_factory=list, description="Detection confidence scores")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "boxes": [[50.0, 100.0, 400.0, 200.0], [50.0, 250.0, 400.0, 500.0]],
                    "labels": ["title", "table"],
                    "scores": [0.96, 0.94],
                }
            ]
        }
    }


class TableResult(BaseModel):
    """Table recognition result"""
    bbox: list[float] = Field(default_factory=list, description="Table bounding box")
    html: str | None = Field(None, description="Table content as HTML")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bbox": [50.0, 250.0, 400.0, 500.0],
                    "html": "<table><tr><th>Item</th><th>Price</th></tr><tr><td>Widget</td><td>$10</td></tr></table>",
                }
            ]
        }
    }


class LayoutPageResult(BaseModel):
    """Layout analysis result for a single page/image"""
    input_path: str | None = Field(None, description="Input file path")
    page_index: int | None = Field(None, description="Page index for multi-page documents")
    layout_det_res: LayoutDetectionResult | None = Field(None, description="Layout detection results")
    table_res: list[TableResult] = Field(default_factory=list, description="Table recognition results")
    text_paragraphs: list[dict[str, Any]] = Field(default_factory=list, description="Text paragraphs with content")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_path": None,
                    "page_index": 0,
                    "layout_det_res": {
                        "boxes": [[50.0, 100.0, 400.0, 200.0], [50.0, 250.0, 400.0, 500.0]],
                        "labels": ["title", "table"],
                        "scores": [0.96, 0.94],
                    },
                    "table_res": [
                        {
                            "bbox": [50.0, 250.0, 400.0, 500.0],
                            "html": "<table><tr><th>Item</th><th>Price</th></tr><tr><td>Widget</td><td>$10</td></tr></table>",
                        }
                    ],
                    "text_paragraphs": [
                        {"text": "Product Catalog", "bbox": [50.0, 100.0, 400.0, 200.0]}
                    ],
                }
            ]
        }
    }


class APIInfoResponse(BaseModel):
    """Root endpoint response with API information"""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="Service description")
    endpoints: dict[str, str] = Field(..., description="Available endpoints")

# Initialize OCR engine
ocr_engine = None

layout_engine = None

def get_ocr_engine():
    """Lazy initialization of PaddleOCR engine for text recognition"""
    global ocr_engine
    if ocr_engine is None:
        logger.info("Initializing PaddleOCR engine...")
        ocr_engine = PaddleOCR(
            lang=LANG,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        logger.info("PaddleOCR engine initialized successfully!")
    return ocr_engine


def get_layout_engine():
    """Lazy initialization of PPStructureV3 engine for layout analysis"""
    global layout_engine
    if layout_engine is None:
        logger.info("Initializing PPStructureV3 layout engine...")
        layout_engine = PPStructureV3(
            lang=LANG,
            use_formula_recognition=False,
            use_table_recognition=True,
            use_textline_orientation=False,
            use_chart_recognition=False,
        )
        logger.info("PPStructureV3 layout engine initialized successfully!")
    return layout_engine

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="PaddleOCR Server")


@app.post("/ocr", response_model=list[OCRPageResult])
async def perform_ocr(request: OCRRequest) -> list[OCRPageResult]:
    """
    Perform OCR text extraction on an image.

    Returns detected text with bounding boxes and confidence scores.
    """
    try:
        engine = get_ocr_engine()
        img = load_image_from_request(request.file)
        result = engine.predict(img)

        results = []
        for r in result:
            raw = r._to_json()
            # PaddleOCR wraps result in 'res' key
            data = raw.get("res", raw)
            results.append(OCRPageResult(
                input_path=data.get("input_path"),
                page_index=data.get("page_index"),
                rec_texts=data.get("rec_texts", []),
                rec_scores=data.get("rec_scores", []),
                rec_polys=data.get("rec_polys", []),
                rec_boxes=data.get("rec_boxes", []),
            ))
        return results
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise


@app.post("/layout", response_model=list[LayoutPageResult])
async def perform_layout(request: OCRRequest) -> list[LayoutPageResult]:
    """
    Perform document layout analysis with table recognition.

    Returns layout regions, tables (as HTML), and text paragraphs.
    """
    try:
        engine = get_layout_engine()
        img = load_image_from_request(request.file)
        result = engine.predict(img)
        return [LayoutPageResult(**x._to_json()) for x in result]
    except Exception as e:
        logger.error(f"Layout processing failed: {str(e)}")
        raise

def load_image_from_request(file_data: str) -> np.ndarray:
    """Load image from base64 string or URL.

    Args:
        file_data: Base64-encoded image data or HTTP(S) URL

    Returns:
        Image as numpy array

    Raises:
        HTTPException: If image download fails (404, timeout, etc.)
    """
    if file_data.startswith(("http://", "https://")):
        try:
            response = requests.get(file_data, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 502
            if status_code == 404:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image not found at URL: {file_data}"
                )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image: HTTP {status_code}"
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=400,
                detail=f"Timeout downloading image from URL: {file_data}"
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image: {str(e)}"
            )
        image = Image.open(io.BytesIO(response.content))
    else:
        try:
            image_data = base64.b64decode(file_data)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {str(e)}"
            )

    return np.array(image)

@app.get("/", response_model=APIInfoResponse)
async def root() -> APIInfoResponse:
    """Root endpoint with API information"""
    return APIInfoResponse(
        service="PaddleOCR Server",
        version="1.0.0",
        description="OCR server with text extraction and layout analysis (PaddleOCR v3+)",
        endpoints={
            "/health": "GET - Health check",
            "/ocr": "POST - Perform OCR text extraction",
            "/layout": "POST - Perform layout analysis with table recognition",
            "/docs": "GET - OpenAPI documentation (Swagger UI)",
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
    )
