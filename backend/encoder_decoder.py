from keras import layers, models, Sequential
from keras.losses import MeanSquaredError


# Load the model
autoencoder = models.load_model('./models/autoencoder-v1.keras', custom_objects={'mse': MeanSquaredError()})

# Separate encoder from decoder
encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('sequential').output)
decoder = models.Model(inputs=autoencoder.get_layer('sequential_1').input, outputs=autoencoder.get_layer('sequential_1').output)

def encode(image):
    return encoder(image)

def decode(latent):
    return decoder(latent)