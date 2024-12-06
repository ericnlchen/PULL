from keras.utils import image_dataset_from_directory
import requests, zipfile, glob
import numpy as np
from keras import layers, models, Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import cv2
import os
import time
import matplotlib.pyplot as plt

from ssim import SSIMLoss, SSIM_MSE_Loss
from attention import AttentionLayer

IMAGE_SIZE = 64
CELEBA_INITIAL_SHAPE = (218, 178)

# Define paths
# dataset_path = './data/tiny-imagenet-200/train'
dataset_path = './data/img_align_celeba'
# Load train dataset
train_dataset = image_dataset_from_directory(
    dataset_path,
    label_mode=None,  # Autoencoder doesn't need labels
    image_size=CELEBA_INITIAL_SHAPE, # initial size for celebA
    batch_size=64,
    shuffle=True
)

def crop_square(image):
    # Assuming the image is taller than it is wide
    shape = image.shape
    print(shape)
    h, w = shape[1], shape[2]
    offset_h = (h - w) // 2
    cropped_image = tf.image.crop_to_bounding_box(image, offset_h, 0, w, w)
    return tf.image.resize(cropped_image, [IMAGE_SIZE, IMAGE_SIZE])

train_dataset = train_dataset.map(lambda x: crop_square(x))
train_dataset = train_dataset.map(lambda x: x / 255.0)

# Encoder
def build_encoder():
    encoder = Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        # 64 x 64

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
        # 32 x 32

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
        # 16 x 16

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
        # 8 x 8

        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        # 8 x 8

    ], name="ae_encoder")
    return encoder

# Decoder
def build_decoder():
    decoder = Sequential([
        layers.Input(shape=(int(IMAGE_SIZE / 8), int(IMAGE_SIZE / 8), 16,)),
        # 8 x 8 x 16

        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(16, (2, 2), strides=2, activation='relu', padding='same'),
        # 16 x 16 x 16

        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(16, (2, 2), strides=2, activation='relu', padding='same'),
        # 32 x 32 x 16

        layers.Conv2D(3, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(3, (2, 2), strides=2, activation='relu', padding='same'),
        # 64 x 64 x 3

        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),

    ], name="ae_decoder")
    return decoder

# Build encoder and decoder
encoder = build_encoder()
decoder = build_decoder()

# Input for the autoencoder
autoencoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

# Combine into a model
autoencoder = models.Model(inputs=autoencoder_input, outputs=decoded)
optimizer = Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss=SSIM_MSE_Loss)
autoencoder.summary()
autoencoder.get_layer('ae_encoder').summary()
autoencoder.get_layer('ae_decoder').summary()

# Test inference time
x = img_to_array(load_img('./data/me-test.jpg')).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
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

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='{epoch:02d}-{loss:.4f}.keras',  # Filepath to save the model
    save_best_only=False,             # Only save the model with the best validation accuracy
    monitor='loss',              # Monitor validation loss (or another metric)
    save_weights_only=False,         # Save the entire model (default: False)
    verbose=1                        # Print a message when a model is saved
)

# Train the autoencoder
history = autoencoder.fit(
    train_dataset,
    epochs=10,
    callbacks=[checkpoint_callback]
)

autoencoder.save('autoencoder-v2.keras')