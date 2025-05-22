FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    nginx \
    python3-pip \
    wget \
    git \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    torch \
    torchvision \
    opencv-python-headless \
    numpy \
    requests \
    sagemaker-pytorch-inference

RUN mkdir -p /opt/program
WORKDIR /opt/program

RUN git clone https://github.com/ctf05/Depth-Anything-V2.git /opt/program/Depth-Anything-V2
WORKDIR /opt/program/Depth-Anything-V2
RUN pip install -r requirements.txt
WORKDIR /opt/program

RUN mkdir -p /opt/program/checkpoints
RUN wget -O /opt/program/checkpoints/depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

COPY predictor.py wsgi.py nginx.conf /opt/program/

COPY serve /opt/program/
RUN sed -i 's/\r$//' /opt/program/serve && \
    chmod +x /opt/program/serve

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

ENTRYPOINT ["/opt/program/serve"]