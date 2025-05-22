from flask import Flask, Response
import flask
import json
import cv2
import numpy as np
import requests
import sys
import os
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))
import torch

app = Flask(__name__)

_model = None

def load_model():
    global _model
    if _model is None:
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        checkpoint_path = "/opt/program/checkpoints/depth_anything_v2_vits.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        _model = model
    return _model

def create_response(status_code: int, body: Dict) -> Dict:
    return {
        'statusCode': status_code,
        'body': json.dumps(body)
    }

def download_from_url(url: str) -> bytes:
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def upload_to_url(url: str, data: bytes, content_type: str):
    response = requests.put(
        url,
        data=data,
        headers={'Content-Type': content_type}
    )
    response.raise_for_status()

@app.route('/ping', methods=['GET'])
def ping():
    return create_response(200, {'status': 'healthy'})

@app.route('/invocations', methods=['POST'])
def invoke():
    if not flask.request.content_type == 'application/json':
        return create_response(415, {
            'error': 'This predictor only supports JSON data'
        })

    model = load_model()

    try:
        request_body = flask.request.get_json()

        read_url = request_body.get('url')
        write_url = request_body.get('write_url')

        if not read_url or not write_url:
            raise ValueError("Missing 'url' or 'write_url' in request body")

        try:
            image_bytes = download_from_url(read_url)
        except Exception as e:
            raise ValueError(f"Failed to download from URL: {str(e)}")

        input_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if input_image is None:
            raise ValueError("Could not decode image from URL")

        original_size = (input_image.shape[1], input_image.shape[0])
        resized_image = cv2.resize(input_image, (400, 400))

        with torch.no_grad():
            depth_map = model.infer_image(resized_image)

        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        resized_depth_map = cv2.resize(depth_map, original_size)
        resized_depth_map = cv2.normalize(resized_depth_map, None, 0, 255, cv2.NORM_MINMAX)
        resized_depth_map = resized_depth_map.astype(np.uint8)

        _, buffer = cv2.imencode('.png', resized_depth_map)

        try:
            upload_to_url(write_url, buffer.tobytes(), 'image/png')
        except Exception as e:
            raise ValueError(f"Failed to upload to write URL: {str(e)}")

        response_body = {
            'success': True,
            'message': 'Depth map generated and uploaded successfully'
        }

        return create_response(200, response_body)

    except Exception as e:
        return create_response(400, {
            'error': str(e)
        })