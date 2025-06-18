# Use official Python slim image
FROM python:3.11-slim

WORKDIR /app


# Install git dan dependensi tambahan (untuk OpenCV dll)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libsm6 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port
EXPOSE 8000

# Set PYTHONPATH agar FastAPI bisa menemukan modul
ENV PYTHONPATH=/app

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]