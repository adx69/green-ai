import argparse
import numpy as np
from PIL import Image
import io
import os
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing_logs.log')
    ]
)
logger = logging.getLogger(__name__)

def process_image_from_file(file_path):
    """Process a single image from a file path"""
    try:
        # Open image from file path
        img = Image.open(file_path)
        logger.info(f"Image opened successfully: {file_path}, Mode: {img.mode}, Size: {img.size}")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info(f"Converted image to RGB mode")
            
        img = img.resize((256, 256))
        logger.info(f"Resized image to 256x256")
        
        return np.array(img) / 255.0
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_images(pre_image_path, post_image_path, job_id):
    """Process the two images and save the results"""
    try:
        logger.info(f"Processing job {job_id}")
        logger.info(f"Pre-image path: {pre_image_path}")
        logger.info(f"Post-image path: {post_image_path}")
        
        # Process images
        logger.info("Processing pre-image...")
        debadihi_post = process_image_from_file(pre_image_path)
        
        logger.info("Processing post-image...")
        debadihi_sw = process_image_from_file(post_image_path)
        
        if debadihi_post is None or debadihi_sw is None:
            logger.error("Failed to process one or both images")
            return False
        
        # Use absolute path for output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
        logger.info(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed numpy arrays with job ID in filenames
        pre_path = os.path.join(output_dir, f"debadihi_post_{job_id}.npy")
        post_path = os.path.join(output_dir, f"debadihi_sw_{job_id}.npy")
        
        logger.info(f"Saving pre-image to: {pre_path}")
        np.save(pre_path, debadihi_post)
        
        logger.info(f"Saving post-image to: {post_path}")
        np.save(post_path, debadihi_sw)
        
        # Verify files were created
        if os.path.exists(pre_path) and os.path.exists(post_path):
            logger.info(f"Files successfully saved. Pre size: {os.path.getsize(pre_path)}, Post size: {os.path.getsize(post_path)}")
            return True
        else:
            logger.error(f"Files not found after saving: {pre_path}, {post_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process pre and post images.')
    parser.add_argument('--preImage', required=True, help='Path to pre-image file')
    parser.add_argument('--postImage', required=True, help='Path to post-image file')
    parser.add_argument('--job_id', required=True, help='Job ID for tracking')
    
    args = parser.parse_args()
    
    # Process the images
    success = process_images(args.preImage, args.postImage, args.job_id)
    
    # Exit with appropriate code
    if success:
        logger.info(f"Job {args.job_id} completed successfully")
        exit(0)
    else:
        logger.error(f"Job {args.job_id} failed")
        exit(1)

if __name__ == '__main__':
    main()