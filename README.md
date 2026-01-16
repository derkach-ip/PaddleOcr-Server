# PaddleOCR HTTP Server

A Docker-based HTTP server for OCR text extraction and document layout analysis using PaddleOCR v3+ with GPU support.

## Features

- **OCR Text Extraction**: Extract text from images with bounding boxes and confidence scores
- **Layout Analysis**: Document structure analysis with table recognition using PPStructureV3
- **GPU Acceleration**: NVIDIA CUDA support for fast inference
- **Docker Ready**: Easy deployment with Docker and GPU passthrough
- **REST API**: Simple HTTP endpoints for integration

## Quick Start

### Build Docker Image

```bash
docker build -t paddleocr-server .
```

### Run Container

```bash
docker run -d \
  --name paddleocr-server \
  --gpus all \
  -p 8080:8080 \
  --restart unless-stopped \
  paddleocr-server
```

### Test Server

```bash
curl http://localhost:8080/health
```

## API Reference

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "PaddleOCR Server"
}
```

### OCR Text Extraction

```
POST /ocr
```

Extract text from an image.

**Request Body:**
```json
{
  "file": "<base64-encoded-image or image-url>",
  "fileType": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Base64-encoded image data or HTTP(S) URL to image |
| `fileType` | int | `0` for PDF, `1` for image (default: 1) |

**Response:**
```json
[
  {
    "input_path": null,
    "page_index": null,
    "rec_texts": ["Hello", "World"],
    "rec_scores": [0.98, 0.95],
    "det_boxes": [
      [[10, 10], [100, 10], [100, 30], [10, 30]],
      [[10, 50], [100, 50], [100, 70], [10, 70]]
    ]
  }
]
```

**Example:**
```bash
# Using URL
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "https://example.com/image.png"}'

# Using base64
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "'$(base64 -w0 image.png)'"}'
```

### Layout Analysis

```
POST /layout
```

Analyze document structure with table recognition.

**Request Body:**
Same as `/ocr` endpoint.

**Response:**
```json
[
  {
    "input_path": null,
    "page_index": null,
    "layout_det_res": {
      "boxes": [...],
      "labels": ["text", "table", "figure"],
      "scores": [0.95, 0.92, 0.88]
    },
    "table_res": [...],
    "text_paragraphs": [...]
  }
]
```

**Example:**
```bash
curl -X POST http://localhost:8080/layout \
  -H "Content-Type: application/json" \
  -d '{"file": "https://example.com/document.png"}'
```

## Configuration

Configure the server using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_HOST` | `0.0.0.0` | Server bind address |
| `OCR_PORT` | `8080` | Server port |
| `OCR_LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `OCR_LANG` | `en` | OCR language (en, ch, japan, korean, etc.) |
| `OCR_CPU_THREADS` | `0` | CPU threads (0 = auto-detect, CPU version only) |
| `OCR_ENABLE_HIGH_INFERENCE` | `false` | Enable HPI (High Performance Inference) with TensorRT |
| `OCR_USE_TENSORRT` | `false` | Enable TensorRT acceleration (GPU only) |
| `OCR_PRECISION` | `fp32` | Inference precision (`fp16` or `fp32`) |

**Example:**
```bash
docker run -d \
  --gpus all \
  -p 9000:9000 \
  -e OCR_PORT=9000 \
  -e OCR_LANG=ch \
  -e OCR_ENABLE_HIGH_INFERENCE=true \
  -e OCR_USE_TENSORRT=true \
  paddleocr-server
```

## Requirements

- Docker with NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA 11.8+ support

## Project Structure

```
├── Dockerfile          # Docker build configuration
├── server.py           # FastAPI server implementation
├── pyproject.toml      # Python dependencies
└── README.md           # This file
```

## Dependencies

- PaddlePaddle GPU 3.2.2+
- PaddleOCR 3.0.0+
- FastAPI
- Uvicorn

## License

MIT License
