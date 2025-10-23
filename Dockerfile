FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY detect.py .
COPY yolov8l.pt .

# Create directory for YOLO models cache
RUN mkdir -p /root/.cache/ultralytics

# Run the detection script
CMD ["python", "-u", "detect.py"]
