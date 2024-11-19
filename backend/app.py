from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import logging
from encoder_decoder import encode, decode

# Configure Flask logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('werkzeug')  # Werkzeug is the Flask server logger
logger.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

@app.route('/api/encode', methods=['GET', 'POST'])
def get_data():
    if request.method == 'POST':
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        
        image = request.files['image']
        image.save(f'uploaded_images/{image.filename}')
        return {'message': f'Image {image.filename} uploaded successfully'}

@app.before_request
def log_request_info():
    logging.info(f"Incoming request: {request.method} {request.path}")
    logging.info(f"Headers: {request.headers}")
    if request.method in ['POST', 'PUT', 'PATCH']:
        try:
            logging.info(f"Body: {request.get_data(as_text=True)}")
        except Exception as e:
            logging.error(f"Error reading request body: {e}")
            
@app.errorhandler(404)
def handle_404_error(e):
    logging.warning(f"Unhandled request: {request.method} {request.path}")
    return {"error": "Not Found"}, 404

if __name__ == '__main__':
    app.run(debug=True)
