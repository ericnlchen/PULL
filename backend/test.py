import numpy as np
from keras import layers, models, Sequential
from keras.losses import MeanSquaredError
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow import reshape
import time

# Load the model
autoencoder = models.load_model('autoencoder.keras', custom_objects={'mse': MeanSquaredError()})


# Call the model
x = img_to_array(load_img('./data/test_img_1.JPEG')).reshape((1, 64, 64, 3))
x = x / 255.0
t0 = time.time()
y = autoencoder(x)
t1 = time.time()
elapsed = t1 - t0
print(f'{elapsed} seconds elapsed')
y = reshape(y, (64, 64, 3))
plt.imshow(y)
plt.show()