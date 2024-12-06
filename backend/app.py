from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import logging
from encode_decode_graph import encode as encode_graph
from encode_decode_graph import decode as decode_graph
from encode_decode_face import encode as encode_face
from encode_decode_face import decode as decode_face
from PIL import Image
import traceback
import numpy as np
from io import BytesIO

# Configure Flask logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('werkzeug')  # Werkzeug is the Flask server logger
logger.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

@app.route('/api/encode-graph', methods=['POST'])
def handle_encode_graph():
    if request.method == 'POST':
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        uploaded_file = request.files['image']
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = image[:, :, 0:3] # remove 4th channel if present
        latent = encode_graph(image)
        latent = latent.tolist()
        return jsonify({'encoded_array': latent})
    
@app.route('/api/decode-graph', methods=['POST'])
def handle_decode_graph():
    if request.method == 'POST':
        data = request.get_json()
        latent = np.array(data.get('array', []))
        recon = decode_graph(latent)
        image = Image.fromarray(recon.astype('uint8'))
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='decodedImage.png')

@app.route('/api/encode-face', methods=['POST'])
def handle_encode_face():
    if request.method == 'POST':
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        uploaded_file = request.files['image']
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = image[:, :, 0:3] # remove 4th channel if present
        latent = encode_face(image)
        latent = latent.tolist()
        return jsonify({'encoded_array': latent})
    
@app.route('/api/decode-face', methods=['POST'])
def handle_decode_face():
    if request.method == 'POST':
        data = request.get_json()
        latent = np.array(data.get('array', []))
        recon = decode_face(latent)
        image = Image.fromarray(recon.astype('uint8'))
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='decodedImage.png')
# @app.before_request
# def log_request_info():
#     logging.info(f"Incoming request: {request.method} {request.path}")
#     logging.info(f"Headers: {request.headers}")
#     if request.method in ['POST', 'PUT', 'PATCH']:
#         try:
#             logging.info(f"Body: {request.get_data(as_text=True)}")
#         except Exception as e:
#             logging.error(f"Error reading request body: {e}")
            
@app.errorhandler(404)
def handle_404_error(e):
    logging.warning(f"Unhandled request: {request.method} {request.path}")
    return {"error": "Not Found"}, 404

@app.errorhandler(Exception)
def handle_exception(e):
    print("Unhandled Exception:", traceback.format_exc())
    return {"error": "Internal Server Error"}, 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
