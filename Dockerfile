FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for PIL and tesseract (optional OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (avoids 2GB+ CUDA deps)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install runtime deps for serving, including doctr for text-density extraction.
RUN pip install --no-cache-dir \
    fastapi==0.135.3 \
    uvicorn==0.44.0 \
    jinja2==3.1.6 \
    jinja2-fragments==1.11.0 \
    python-multipart==0.0.24 \
    python-dotenv==1.1.0 \
    pillow==12.1.1 \
    numpy==2.4.2 \
    matplotlib==3.10.8 \
    scikit-learn==1.8.0 \
    torch-geometric==2.7.0 \
    python-doctr[torch]==1.0.1 \
    pytesseract==0.3.13 \
    huggingface_hub==0.31.0 \
    ultralytics==8.4.37 \
    sentence-transformers==3.4.1 \
    plotly==5.24.1 \
    structlog==25.5.0 \
    seqlog==0.4.3

# Copy application code
COPY src/ src/
COPY app/ app/
COPY conftest.py ./

# Models are volume-mounted at runtime
VOLUME /app/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
