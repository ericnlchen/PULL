import numpy as np
from keras import models
from attention import AttentionLayer
from ssim import SSIMLoss, SSIM_MSE_Loss

# Load the model
autoencoder = models.load_model(
    'models/autoencoder-v2.keras',
    custom_objects={'AttentionLayer': AttentionLayer, 'SSIMLoss': SSIMLoss, 'SSIM_MSE_Loss': SSIM_MSE_Loss}
)

# Separate encoder from decoder
try:
    encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('ae_encoder').output)
    decoder = models.Model(inputs=autoencoder.get_layer('ae_decoder').input, outputs=autoencoder.get_layer('ae_decoder').output)
except ValueError:
    # use old naming scheme
    encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('sequential').output)
    decoder = models.Model(inputs=autoencoder.get_layer('sequential_1').input, outputs=autoencoder.get_layer('sequential_1').output)

def encode(image):
    image = image / 255.0
    image = image.reshape(get_input_shape())
    z = encoder(image)
    return np.array(z).reshape(1, get_latent_length())

def decode(latent):
    latent = latent.reshape(get_latent_shape())
    recon = decoder(latent)
    return np.array(recon).reshape(get_input_shape()).squeeze() * 255.0

def get_latent_length():
    length = 1
    for d in encoder.output.shape:
        if (d is not None):
            length *= d
    return length

def get_latent_shape():
    shape = []
    for d in encoder.output.shape:
        if (d is not None):
            shape.append(d)
        else:
            shape.append(1)
    return tuple(shape)

def get_input_shape():
    shape = []
    for d in encoder.input.shape:
        if (d is not None):
            shape.append(d)
        else:
            shape.append(1)
    return tuple(shape)