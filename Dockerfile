FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up directories
RUN mkdir -p /opt/ml/code /opt/ml/model

# Set the working directory
WORKDIR /opt/ml/code

# Copy requirements.txt
COPY requirements.txt /opt/ml/code/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code directory containing inference.py and depth_anything_v2
COPY code/ /opt/ml/code/

# Create model directory and download the model weights
RUN mkdir -p /opt/ml/model/checkpoints
RUN wget -O /opt/ml/model/checkpoints/depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Expose port 8080 for SageMaker
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["python", "-m", "inference"]