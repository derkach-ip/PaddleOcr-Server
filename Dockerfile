FROM paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6

WORKDIR /workspace

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Copy only dependency file first (changes less frequently)
COPY pyproject.toml /workspace/pyproject.toml

# Install dependencies (cached unless pyproject.toml changes)
# Note: Base image already has paddlepaddle-gpu installed, don't overwrite it
# First, force reinstall PyYAML to handle conflict with base image's distutils version
RUN pip install --ignore-installed PyYAML
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml

# Install HPI dependencies for TensorRT acceleration
# Copy pre-downloaded ultra-infer wheel to avoid slow download during build
COPY ultra_infer_gpu_python-1.2.0-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip install /tmp/ultra_infer_gpu_python-1.2.0-cp310-cp310-linux_x86_64.whl && \
    rm /tmp/ultra_infer_gpu_python-1.2.0-cp310-cp310-linux_x86_64.whl

# Install paddle2onnx for full HPI functionality
# Use CUDA compat library path only during build (provides stub libcuda.so.1)
ENV DISABLE_MODEL_SOURCE_CHECK=True
RUN LD_LIBRARY_PATH="/usr/local/cuda-11.8/compat:${LD_LIBRARY_PATH}" \
    pip install paddle2onnx==2.0.2rc3 onnx==1.17.0 onnxoptimizer==0.3.13 polygraphy>=0.49.20

# Fix TensorRT library path - base image has wrong path (8.5.3.1 vs actual 8.6.1.6)
ENV LD_LIBRARY_PATH="/usr/local/TensorRT-8.6.1.6/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Copy application code last (changes frequently, but doesn't invalidate dependency cache)
COPY server.py /workspace/server.py

EXPOSE 8080

CMD ["python", "/workspace/server.py"]
