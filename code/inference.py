import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'site-packages'))
print(f"Site packages: {os.listdir(os.path.join(os.path.dirname(__file__), 'site-packages'))}")
import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def model_fn(model_dir):
    print(f"Model directory contents: {os.listdir(model_dir)}")
    print(f"Model directory: {model_dir}")
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(f"{model_dir}/checkpoints/depth_anything_v2_vits.pth", map_location='cpu'))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/octet-stream':
        return cv2.imdecode(np.frombuffer(request_body, np.uint8), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        return model.infer_image(input_data)

def output_fn(prediction, accept):
    if accept == 'application/json':
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported accept type: {accept}")