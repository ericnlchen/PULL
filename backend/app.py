from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin
import logging
from encoder_decoder import encode, decode
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

@app.route('/api/encode', methods=['POST'])
def handle_encode():
    if request.method == 'POST':
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        
        uploaded_file = request.files['image']

        image = Image.open(uploaded_file)
        x = np.array(image) / 255.0
        x = x[:, :, 0:3] # remove 4th channel if present
        x = x.reshape((1, 64, 64, 3))
        z = encode(x)
        print(z.shape)
        z = np.array(z).reshape((1, z.shape[1]*z.shape[2]*z.shape[3]))
        z_list = z.tolist()

        return jsonify({'encoded_array': z_list})
    
@app.route('/api/decode', methods=['POST'])
def handle_decode():
    if request.method == 'POST':
        data = request.get_json()
        latent_z = np.array(data.get('array', []))
        latent_z = latent_z.reshape(1, 8, 8, 16)

        y = decode(latent_z)
        y = np.array(y).reshape(64, 64, 3) * 255.0

        image = Image.fromarray(y.astype('uint8'))

        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='decodedImage.png')
        
        # image = Image.open(uploaded_file)
        # x = np.array(image) / 255.0
        # x = x.reshape((1, 64, 64, 3))
        # z = encode(x)

        # z_list = np.array(z).tolist()

        # return jsonify({'encoded_array': z_list})

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
