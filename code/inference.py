import json
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), 'site-packages'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))
import torch
import requests
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def model_fn(model_dir):
    logger.info(f"Starting model initialization with model_dir: {model_dir}")
    try:
        logger.info("Creating DepthAnythingV2 model with encoder='vits', features=64")
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        logger.info("Model created successfully")

        checkpoint_path = f"{model_dir}/checkpoints/depth_anything_v2_vits.pth"
        logger.info(f"Loading model state dict from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        logger.info("Model state dict loaded successfully")

        logger.info("Setting model to evaluation mode")
        model.eval()
        logger.info("Model initialization completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    logger.info(f"Starting input processing with content_type: {request_content_type}")
    logger.debug(f"Request body length: {len(request_body) if request_body else 0}")

    if request_content_type == 'application/json':
        try:
            logger.info("Parsing JSON request body")
            # Parse JSON to get URL
            input_data = json.loads(request_body)
            logger.info("JSON parsed successfully")
            logger.debug(f"Parsed input data keys: {list(input_data.keys())}")

            # Extract the URLs
            read_url = input_data.get('url')
            write_url = input_data.get('write_url')
            logger.info(f"Extracted read_url: {read_url}")
            logger.info(f"Extracted write_url: {write_url}")

            if not read_url or not write_url:
                logger.error("Missing required URL fields in input")
                raise ValueError("Missing 'url' or 'write_url' field in input JSON.")

            logger.info(f"Downloading image from URL: {read_url}")
            # Download image from the read URL
            response = requests.get(read_url)
            logger.info(f"HTTP response status: {response.status_code}")
            response.raise_for_status()
            logger.info("Image downloaded successfully")

            # Convert to OpenCV image
            image_bytes = response.content
            logger.info(f"Downloaded image size: {len(image_bytes)} bytes")
            logger.info("Converting bytes to OpenCV image")
            input_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

            if input_image is None:
                logger.error("Failed to decode image from downloaded bytes")
                raise ValueError("Could not decode image from URL")

            logger.info(f"Image decoded successfully, shape: {input_image.shape}")

            original_size = (input_image.shape[1], input_image.shape[0]) # (width, height)
            logger.info(f"Original image size: {original_size}")

            # Resize to 400x400
            logger.info("Resizing image to 400x400")
            resized_image = cv2.resize(input_image, (400, 400))
            logger.info(f"Image resized successfully, new shape: {resized_image.shape}")

            logger.info("Input processing completed successfully")
            return resized_image, original_size, write_url
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error while downloading image: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in input_fn: {str(e)}")
            raise
    else:
        logger.error(f"Unsupported content type received: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    logger.info("Starting prediction")
    try:
        resized_image, original_size, write_url = input_data
        logger.info(f"Input data - image shape: {resized_image.shape}, original_size: {original_size}")
        logger.debug(f"Write URL: {write_url}")

        logger.info("Running model inference with torch.no_grad()")
        with torch.no_grad():
            logger.info("Calling model.infer_image()")
            depth_map = model.infer_image(resized_image)
            logger.info(f"Inference completed, depth_map type: {type(depth_map)}")
            if hasattr(depth_map, 'shape'):
                logger.info(f"Depth map shape: {depth_map.shape}")

        logger.info("Prediction completed successfully")
        return depth_map, original_size, write_url
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, accept):
    logger.info(f"Starting output processing with accept type: {accept}")
    try:
        depth_map, original_size, write_url = prediction
        logger.info(f"Processing depth map of type: {type(depth_map)}")
        logger.info(f"Target original size: {original_size}")

        if accept == 'application/json':
            logger.info("Processing for JSON output")

            if isinstance(depth_map, torch.Tensor):
                logger.info("Converting torch tensor to numpy array")
                depth_map = depth_map.cpu().numpy()
                logger.info(f"Converted to numpy, shape: {depth_map.shape}")

            # Resize depth map back to the original input image size
            logger.info(f"Resizing depth map back to original size: {original_size}")
            resized_depth_map = cv2.resize(depth_map, original_size)
            logger.info(f"Resized depth map shape: {resized_depth_map.shape}")

            # Normalize the depth map for visualization
            logger.info("Normalizing depth map for visualization")
            resized_depth_map = cv2.normalize(resized_depth_map, None, 0, 255, cv2.NORM_MINMAX)
            logger.info("Normalization completed")

            logger.info("Converting to uint8")
            resized_depth_map = resized_depth_map.astype(np.uint8)
            logger.info(f"Final depth map shape: {resized_depth_map.shape}, dtype: {resized_depth_map.dtype}")

            # Encode resized depth map as PNG
            logger.info("Encoding depth map as PNG")
            _, buffer = cv2.imencode('.png', resized_depth_map)
            logger.info(f"PNG encoding completed, buffer size: {len(buffer)} bytes")

            # Upload to the provided write URL
            logger.info(f"Uploading to write URL: {write_url}")
            response = requests.put(
                write_url,
                data=buffer.tobytes(),
                headers={'Content-Type': 'image/png'}
            )
            logger.info(f"Upload response status: {response.status_code}")
            response.raise_for_status()
            logger.info("Upload completed successfully")

            result = json.dumps({'success': True, 'message': 'Depth map generated and uploaded successfully'})
            logger.info("Output processing completed successfully")
            return result
        else:
            logger.error(f"Unsupported accept type: {accept}")
            raise ValueError(f"Unsupported accept type: {accept}")
    except requests.RequestException as e:
        logger.error(f"Request error during upload: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in output_fn: {str(e)}")
        raise