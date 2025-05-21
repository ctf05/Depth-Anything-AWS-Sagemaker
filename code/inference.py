import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'site-packages'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))
import torch
import requests
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def model_fn(model_dir):
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(f"{model_dir}/checkpoints/depth_anything_v2_vits.pth", map_location='cpu'))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        # Parse JSON to get URL
        input_data = json.loads(request_body)

        # Extract the URLs
        read_url = input_data.get('url')
        write_url = input_data.get('write_url')

        if not read_url or not write_url:
            raise ValueError("Missing 'url' or 'write_url' field in input JSON.")

        # Download image from the read URL
        response = requests.get(read_url)
        response.raise_for_status()

        # Convert to OpenCV image
        image_bytes = response.content
        input_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        original_size = (input_image.shape[1], input_image.shape[0])  # (width, height)
        # Resize to 400x400
        resized_image = cv2.resize(input_image, (400, 400))

        return resized_image, original_size, write_url
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    resized_image, original_size, write_url = input_data
    with torch.no_grad():
        depth_map = model.infer_image(resized_image)
    return depth_map, original_size, write_url

def output_fn(prediction, accept):
    depth_map, original_size, write_url = prediction
    if accept == 'application/json':
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        # Resize depth map back to the original input image size
        resized_depth_map = cv2.resize(depth_map, original_size)
        # Normalize the depth map for visualization
        resized_depth_map = cv2.normalize(resized_depth_map, None, 0, 255, cv2.NORM_MINMAX)
        resized_depth_map = resized_depth_map.astype(np.uint8)

        # Encode resized depth map as PNG
        _, buffer = cv2.imencode('.png', resized_depth_map)

        # Upload to the provided write URL
        response = requests.put(
            write_url,
            data=buffer.tobytes(),
            headers={'Content-Type': 'image/png'}
        )
        response.raise_for_status()

        return json.dumps({'success': True, 'message': 'Depth map generated and uploaded successfully'})
    raise ValueError(f"Unsupported accept type: {accept}")