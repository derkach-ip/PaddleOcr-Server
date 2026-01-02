# PaddleOCR HTTP Server

A GPU-accelerated HTTP server for OCR text extraction and document layout analysis using PaddleOCR v3+ and PPStructureV3.

## Quick Start

### GPU Version
```bash
docker run -d \
  --name paddleocr-server \
  --gpus all \
  -p 8080:8080 \
  jarvis1tube/paddleocr-server:gpu
```

### CPU Version
```bash
docker run -d \
  --name paddleocr-server \
  -p 8080:8080 \
  jarvis1tube/paddleocr-server:cpu
```

Test it:
```bash
curl http://localhost:8080/health
```

## Features

- **OCR Text Extraction** - Extract text with bounding boxes and confidence scores
- **Layout Analysis** - Document structure analysis with table recognition (PPStructureV3)
- **GPU Acceleration** - NVIDIA CUDA 12.9 support for fast inference
- **REST API** - Simple JSON endpoints for easy integration
- **Multiple Formats** - Supports base64-encoded images and image URLs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ocr` | POST | Text extraction |
| `/layout` | POST | Layout analysis with table recognition |

### OCR Example

```bash
# Using image URL
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "https://example.com/image.png"}'

# Using base64
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "'$(base64 -w0 image.png)'"}'
```

### Response

```json
[
  {
    "rec_texts": ["Hello", "World"],
    "rec_scores": [0.98, 0.95],
    "rec_polys": [[[10, 10], [100, 10], [100, 30], [10, 30]]]
  }
]
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_HOST` | `0.0.0.0` | Server bind address |
| `OCR_PORT` | `8080` | Server port |
| `OCR_LOG_LEVEL` | `info` | Log level (debug, info, warning, error) |
| `OCR_LANG` | `en` | OCR language (en, ch, japan, korean, etc.) |
| `OCR_CPU_THREADS` | `0` | CPU threads to use (0 = auto-detect, CPU version only) |

### Custom Configuration

```bash
docker run -d \
  --gpus all \
  -p 9000:9000 \
  -e OCR_PORT=9000 \
  -e OCR_LANG=ch \
  jarvis1tube/paddleocr-server:gpu
```

## Requirements

**GPU version:**
- NVIDIA GPU with CUDA 12.9+ support
- Docker with NVIDIA Container Toolkit

**CPU version:**
- Docker only (no GPU required)

## Tags

- `latest`, `gpu` - GPU version with CUDA 12.9 support
- `cpu` - CPU version (no GPU required)

## Links

- [GitHub Repository](https://github.com/derkach-ip/PaddleOcr-Server)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## License

MIT License
