from keras.utils import image_dataset_from_directory
import requests, zipfile, glob
import numpy as np
from keras import layers, models, Sequential
from keras.preprocessing.image import load_img, img_to_array
import cv2
import os
import time

# Define paths
dataset_path = './data/tiny-imagenet-200/train'

# Load train dataset
train_dataset = image_dataset_from_directory(
    dataset_path,
    label_mode=None,  # Autoencoder doesn't need labels
    image_size=(64, 64),
    batch_size=64,
    shuffle=True
)

train_dataset = train_dataset.map(lambda x: x / 255.0)

# Encoder
def build_encoder():
    encoder = Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu')
    ])
    return encoder

# Decoder
def build_decoder():
    decoder = Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(8 * 8 * 128, activation='relu'),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return decoder

# Build encoder and decoder
encoder = build_encoder()
decoder = build_decoder()

# Input for the autoencoder
autoencoder_input = layers.Input(shape=(64, 64, 3))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

# Combine into a model
autoencoder = models.Model(inputs=autoencoder_input, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Test inference time
x = img_to_array(load_img('./data/test_img_1.JPEG')).reshape((1, 64, 64, 3))
x = x / 255.0
y = autoencoder(x) # warmup
t0 = time.time()
y = autoencoder(x)
t1 = time.time()
print(f'{t1 - t0} seconds elapsed')

# for batch in train_dataset.take(1):
#     print(f"Batch shape: {batch.shape}")
#     print(f"Batch dtype: {batch.dtype}")
#     print(f"Batch example: {batch[0].shape}")

# Set labels equal to training examples
train_dataset = train_dataset.map(lambda x: (x, x))

# Train the autoencoder
autoencoder.fit(
    train_dataset,
    epochs=20
)

autoencoder.save('autoencoder-v2.keras')