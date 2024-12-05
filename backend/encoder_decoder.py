from keras import layers, models, Sequential
from keras.losses import MeanSquaredError
from attention import AttentionLayer
from ssim import SSIMLoss

# Load the model
autoencoder = models.load_model('models/autoencoder-v2.keras', custom_objects={'AttentionLayer': AttentionLayer, 'SSIMLoss': SSIMLoss})

# Separate encoder from decoder
encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('ae_encoder').output)
decoder = models.Model(inputs=autoencoder.get_layer('ae_decoder').input, outputs=autoencoder.get_layer('ae_decoder').output)

def encode(image):
    return encoder(image)

def decode(latent):
    return decoder(latent)