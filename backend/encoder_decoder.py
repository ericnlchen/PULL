from keras import layers, models, Sequential
from keras.losses import MeanSquaredError
from attention import AttentionLayer
from ssim import SSIMLoss

# Load the model
autoencoder = models.load_model(
    'models/autoencoder-v1.keras',
    custom_objects={'AttentionLayer': AttentionLayer, 'SSIMLoss': SSIMLoss}
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
    return encoder(image)

def decode(latent):
    return decoder(latent)

def get_latent_length():
    length = 1
    for d in encoder.output.shape:
        if (d is not None):
            length += d
    return d

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

def get_input_shape_no_batch():
    shape = []
    for d in encoder.input.shape:
        if (d is not None):
            shape.append(d)
    return tuple(shape)