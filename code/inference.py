import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'site-packages'))
import torch
import base64
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
        # Parse JSON to get base64 string
        input_data = json.loads(request_body)
        if 'body' not in input_data or 'image' not in input_data['body']:
            raise ValueError("Missing 'body' or 'image' field in input JSON.")

        # Extract the base64 image string
        base64_image = input_data['body']['image']
        # Decode base64 string to image
        image_bytes = base64.b64decode(base64_image)
        input_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        original_size = (input_image.shape[1], input_image.shape[0])  # (width, height)
        # Resize to 400x400
        resized_image = cv2.resize(input_image, (400, 400))
        return resized_image, original_size
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    resized_image, original_size = input_data
    with torch.no_grad():
        depth_map = model.infer_image(resized_image)
    return depth_map, original_size

def output_fn(prediction, accept):
    depth_map, original_size = prediction
    if accept == 'application/json':
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        # Resize depth map back to the original input image size
        resized_depth_map = cv2.resize(depth_map, original_size)
        # Normalize the depth map for visualization (optional, depends on the use case)
        resized_depth_map = cv2.normalize(resized_depth_map, None, 0, 255, cv2.NORM_MINMAX)
        resized_depth_map = resized_depth_map.astype(np.uint8)
        # Encode resized depth map as PNG and then convert to base64
        _, buffer = cv2.imencode('.png', resized_depth_map)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        # Return the base64 encoded depth map in the same format as input, but with 'depthmap' instead of 'image'
        return json.dumps({'body': {'depthmap': base64_image}})
    raise ValueError(f"Unsupported accept type: {accept}")
