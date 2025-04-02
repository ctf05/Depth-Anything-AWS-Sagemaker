FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    sagemaker-pytorch-inference \
    matplotlib \
    scikit-image \
    opencv-python \
    pillow

# Set up directories
RUN mkdir -p /opt/ml/code /opt/ml/model

# Set the working directory
WORKDIR /opt/ml/code

# Clone depth anything repository
RUN git clone https://github.com/LiheYoung/Depth-Anything /opt/ml/code/depth-anything
WORKDIR /opt/ml/code/depth-anything
# Install depth-anything
RUN pip install -r requirements.txt

WORKDIR /opt/ml/code

# Copy code directory containing inference.py
COPY code/ /opt/ml/code/

# Create model directory and download the model weights
RUN mkdir -p /opt/ml/model/checkpoints
RUN wget --header="Authorization: bearer ${HF_TOKEN}" -O /opt/ml/model/checkpoints/depth_anything_v2_vits.pth https://huggingface.co/shariqfarooq/Depth-Anything/resolve/main/checkpoints/depth_anything_v2_vits.pth

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Expose port 8080 for SageMaker
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["python", "-m", "inference"]