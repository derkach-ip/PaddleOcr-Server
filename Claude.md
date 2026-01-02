# PaddleOCR HTTP Server - Development Notes

## Project Overview
Docker-based PaddleOCR v3+ HTTP server with GPU support (NVIDIA CUDA 12.9) for OCR text extraction and document layout analysis.

## Architecture

### Docker Image
- **Base**: `paddlepaddle/paddle:3.2.2-gpu-cuda12.9-cudnn9.9`
- **Package Manager**: `uv` for fast dependency resolution
- **Python**: 3.10

### Server
- **Framework**: FastAPI + Uvicorn
- **Engines**: PaddleOCR (text) + PPStructureV3 (layout/tables)
- **Lazy Init**: Engines load on first request

### Endpoints
- `GET /` - API info
- `GET /health` - Health check
- `POST /ocr` - Text extraction
- `POST /layout` - Layout analysis with table recognition

## Configuration

Environment variables:
- `OCR_HOST` - Server host (default: 0.0.0.0)
- `OCR_PORT` - Server port (default: 8080)
- `OCR_LOG_LEVEL` - Log level (default: info)
- `OCR_LANG` - OCR language (default: en)

## Build & Run

```bash
# Build
docker build -t paddleocr-server .

# Run
docker run -d --name paddleocr-server --gpus all -p 8080:8080 --restart unless-stopped paddleocr-server

# Test
curl http://localhost:8080/health
```

## Issues Resolved
1. PaddlePaddle GPU not on PyPI - Install from unofficial Chinese repository
2. PaddleX serving plugin unavailable - Created custom FastAPI server
3. Deprecated OCR parameters - Updated to use_textline_orientation, use_doc_orientation_classify
