from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
import traceback
import threading
import subprocess
import time
import torch
import numpy as np
from unet_model import UNet
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('main_controller_logs.log')
    ]
)
logger = app.logger

# Track job status
job_status = {}

# Load the UNet model
def load_model():
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "unet_model.pth")
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}, creating a new model")
            
            # Create a new model instance
            model = UNet(in_channels=3, out_channels=1)
            
            # Save the initial model
            torch.save(model.state_dict(), model_path)
            logger.info(f"Created and saved new model to {model_path}")
        else:
            logger.info(f"Loading model from {model_path}")
            
            # Create model instance
            model = UNet(in_channels=3, out_channels=1)
            
            # Load saved weights
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Add this function after the load_model function

def ensure_model_loaded():
    """Makes sure the model is loaded, attempting to reload if necessary"""
    global model
    if model is None:  # Changed from "is not None" to "is None"
        logger.info("Model not loaded, attempting to load again")
        model = load_model()
    return model is not None

# Add this function after ensure_model_loaded

def initialize_model_if_needed():
    """Creates and trains a simple UNet model if none exists"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "unet_model.pth")
    
    if not os.path.exists(model_path):
        logger.info("Model not found, running training script to create one")
        success = run_training_script()
        if not success:
            logger.error("Failed to create initial model through training")
            
            # Create a fallback model - just initialize weights but don't train
            logger.info("Creating untrained model as fallback")
            model = UNet(in_channels=3, out_channels=1)
            
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the untrained model
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved untrained model to {model_path}")

# Global model instance
model = load_model()

def save_uploaded_files(files):
    """Save uploaded files to a temporary location"""
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_paths = {}
    for name, file in files.items():
        # Create a unique filename to avoid collisions
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}_{file.filename}"
        filepath = os.path.join(temp_dir, filename)
        
        # Save the file
        file.save(filepath)
        logger.info(f"Saved {name} to {filepath}")
        saved_paths[name] = filepath
    
    return saved_paths

def run_training_script():
    """Run the model training script"""
    try:
        logger.info("Starting model training script")
        
        # Get the path to the train_model.py script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_model.py")
        
        # Run the training script as a subprocess
        cmd = ["python", script_path]
        
        logger.info(f"Executing training script: {' '.join(cmd)}")
        
        # Start the subprocess with real-time output handling
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Process output in real-time
        logger.info("Training progress:")
        
        # Read and log stdout in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"TRAIN: {line}")
        
        # Get return code
        process.wait()
        
        # Check for errors in stderr
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"Training stderr: {stderr}")
        
        if process.returncode == 0:
            logger.info("Training script completed successfully")
            return True
        else:
            logger.error(f"Training script failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error running training script: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def process_images_async(job_id, file_paths):
    """Run the preprocessing script asynchronously"""
    try:
        job_status[job_id] = {"status": "processing", "start_time": time.time()}
        
        # Get the path to the pre-process.py script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre-process.py")
        
        # Run the preprocessing script as a subprocess
        cmd = [
            "python", 
            script_path, 
            "--preImage", file_paths["preImage"],
            "--postImage", file_paths["postImage"],
            "--job_id", job_id
        ]
        
        logger.info(f"Starting preprocessing job {job_id}: {' '.join(cmd)}")
        
        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Capture output
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Job {job_id} preprocessing completed successfully")
            
            # Run model inference
            result_path = run_model_inference(job_id)
            
            if result_path:
                # Update job status
                job_status[job_id] = {
                    "status": "running_inference",
                    "message": "UNet inference completed, running training script"
                }
                
                # Run the training script after successful inference
                training_success = run_training_script()
                
                if training_success:
                    job_status[job_id] = {
                        "status": "completed",
                        "end_time": time.time(),
                        "message": "Processing and training completed successfully",
                        "result_path": result_path
                    }
                else:
                    job_status[job_id] = {
                        "status": "completed_with_warnings",
                        "end_time": time.time(),
                        "message": "Processing completed but training failed",
                        "result_path": result_path
                    }
            else:
                job_status[job_id] = {
                    "status": "failed",
                    "end_time": time.time(),
                    "error": "Model inference failed"
                }
        else:
            logger.error(f"Job {job_id} failed: {stderr.decode()}")
            job_status[job_id] = {
                "status": "failed",
                "end_time": time.time(),
                "error": stderr.decode()
            }
            
    except Exception as e:
        logger.error(f"Error in async processing for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        job_status[job_id] = {
            "status": "failed",
            "end_time": time.time(),
            "error": str(e)
        }

def run_model_inference(job_id):
    """Run the UNet model on preprocessed data"""
    try:
        if not ensure_model_loaded():
            logger.error("Failed to load model, cannot run inference")
            return None
            
        # Get paths to the preprocessed data
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
        pre_path = os.path.join(processed_dir, f"debadihi_post_{job_id}.npy")
        post_path = os.path.join(processed_dir, f"debadihi_sw_{job_id}.npy")
        
        # Check if files exist
        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            logger.error(f"Preprocessed data not found: {pre_path}, {post_path}")
            return None
            
        # Load the preprocessed data
        pre_image = np.load(pre_path)
        post_image = np.load(post_path)
        
        logger.info(f"Pre-image shape: {pre_image.shape}, Post-image shape: {post_image.shape}")
        
        # OPTION 1: Use only post_image (3 channels)
        input_data = post_image
        
        # Prepare input tensor
        input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).float().unsqueeze(0)
        
        logger.info(f"Running model inference for job {job_id}, input shape: {input_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Process output
        result = output.squeeze().cpu().numpy()
        
        # Save the result
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save as both numpy array and image
        result_path = os.path.join(results_dir, f"result_{job_id}")
        np_path = f"{result_path}.npy"
        img_path = f"{result_path}.png"
        
        # Save numpy array
        np.save(np_path, result)
        
        # Process result for visualization
        # Apply sigmoid to get probability map (0-1)
        result_sigmoid = 1 / (1 + np.exp(-result))
        
        # Normalize to 0-255 range
        result_img = (result_sigmoid * 255).astype(np.uint8)
        
        # Create a proper RGB visualization
        # Create colored visualization (green intensity shows prediction)
        h, w = result_img.shape
        rgb_result = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_result[:, :, 1] = result_img  # Green channel shows predictions
        
        # Overlay on original image for context
        original_img = (post_image * 255).astype(np.uint8)
        alpha = 0.6  # Transparency factor
        visualization = (alpha * original_img + (1 - alpha) * rgb_result).astype(np.uint8)
        
        # Save as visualization image
        Image.fromarray(visualization).save(img_path)
        
        logger.info(f"Inference completed and results saved to {img_path}")
        return img_path
        
    except Exception as e:
        logger.error(f"Error running model inference: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/', methods=['POST'])
def handle_upload():
    try:
        logger.info("Received image processing request")
        
        # Check if files are in the request
        if 'preImage' not in request.files or 'postImage' not in request.files:
            logger.error("Missing required image files")
            return jsonify({'error': 'Both pre and post images are required'}), 400
            
        # Create a job ID
        job_id = f"job_{int(time.time())}"
        
        # Save the uploaded files
        file_paths = save_uploaded_files(request.files)
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=process_images_async,
            args=(job_id, file_paths)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately with job ID for status tracking
        return jsonify({
            'status': 'processing',
            'message': 'Images received and processing started',
            'job_id': job_id
        })
        
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check the status of a processing job"""
    if job_id in job_status:
        return jsonify(job_status[job_id])
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Return the result image for a completed job"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
        
    job_info = job_status[job_id]
    if job_info.get('status') not in ['completed', 'completed_with_warnings']:
        return jsonify({'error': 'Job not completed or failed'}), 400
        
    result_path = job_info.get('result_path')
    if not result_path or not os.path.exists(result_path):
        return jsonify({'error': 'Result not found'}), 404
        
    return send_file(result_path, mimetype='image/png')

@app.route('/train', methods=['POST'])
def trigger_training():
    """Endpoint to manually trigger model training"""
    try:
        logger.info("Received request to trigger model training")
        
        # Start training in a background thread to avoid blocking
        thread = threading.Thread(target=run_training_script)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Model training started in background'
        })
        
    except Exception as e:
        logger.error(f"Error triggering training: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the model exists and is ready for use
    initialize_model_if_needed()
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting main controller on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)