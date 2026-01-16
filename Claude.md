# PaddleOCR HTTP Server - Development Notes

## Project Overview
Docker-based PaddleOCR v3+ HTTP server with GPU support (NVIDIA CUDA 11.8 + TensorRT 8.6) for OCR text extraction and document layout analysis.

## Architecture

### Docker Image
- **Base**: `paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6`
- **Package Manager**: `uv` for fast dependency resolution
- **Python**: 3.10
- **HPI Support**: ultra-infer for TensorRT acceleration

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
- `OCR_CPU_THREADS` - CPU threads (default: 0, auto-detect)
- `OCR_ENABLE_HIGH_INFERENCE` - Enable HPI with TensorRT (default: false)
- `OCR_USE_TENSORRT` - Enable TensorRT acceleration (default: false)
- `OCR_PRECISION` - Inference precision, fp16 or fp32 (default: fp32)

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
4. TensorRT library path mismatch - Fixed LD_LIBRARY_PATH for TensorRT 8.6.1.6
5. HPI support - Added ultra-infer wheel and paddle2onnx for TensorRT acceleration
