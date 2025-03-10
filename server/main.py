from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

@app.route('/', methods=['POST'])
def process_images():
    try:
        logger.info("Received image processing request")
        
        # Check if files are in the request
        if 'preImage' not in request.files or 'postImage' not in request.files:
            logger.error("Missing required image files")
            return jsonify({'error': 'Both pre and post images are required'}), 400
            
        pre_image = request.files['preImage']
        post_image = request.files['postImage']
        
        # Log image information
        logger.info(f"Pre-image: {pre_image.filename}, Post-image: {post_image.filename}")
        
        
        # Example response 
        result = {
            'status': 'success',
            'message': 'Images received successfully',
            'saplingDetection': {
                'preImage': {
                    'count': 12,
                    'greenCoverage': 42.5
                },
                'postImage': {
                    'count': 18,
                    'greenCoverage': 58.7
                },
                'growthRate': 38.1
            }
        }
        
        logger.info("Processing completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)