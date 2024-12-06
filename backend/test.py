import numpy as np
from keras import layers, models, Sequential
from keras.losses import MeanSquaredError
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from attention import AttentionLayer
from ssim import SSIMLoss, SSIM_MSE_Loss

# Load the model
autoencoder = models.load_model('models/03-0.1987.keras', custom_objects={'AttentionLayer': AttentionLayer, 'SSIMLoss': SSIMLoss, 'SSIM_MSE_Loss': SSIM_MSE_Loss})

# Call the model
x = img_to_array(load_img('./data/me-test.jpg')).reshape((1, 64, 64, 3))
x = x / 255.0
t0 = time.time()
y = autoencoder(x)
t1 = time.time()
elapsed = t1 - t0
print(f'{elapsed} seconds elapsed')
y = tf.reshape(y, (64, 64, 3))
plt.imshow(y)
plt.show()