import numpy as np
from keras import layers, models, Sequential
from keras.losses import MeanSquaredError
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from attention import AttentionLayer
from ssim import SSIMLoss, SSIM_MSE_Loss

import torch
from torchvision.utils import save_image
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) 

# Load the model
# autoencoder = models.load_model('models/03-0.1987.keras', custom_objects={'AttentionLayer': AttentionLayer, 'SSIMLoss': SSIMLoss, 'SSIM_MSE_Loss': SSIM_MSE_Loss})
vae = torch.load('models/vae_model_20.pth', map_location='cpu')

# Call the model
# x = img_to_array(load_img('./data/me-test.jpg')).reshape((1, 64, 64, 3))
# x = x / 255.0
# t0 = time.time()
# y = autoencoder(x)
# t1 = time.time()
# elapsed = t1 - t0
# print(f'{elapsed} seconds elapsed')
# y = tf.reshape(y, (64, 64, 3))
# plt.imshow(y)
# plt.show()
x = img_to_array(load_img('./data/me-test-150.jpg')).astype('float32') / 255.0  # Normalize to [0, 1]
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions to [1, 3, H, W]
mu, log_var = vae.encode(x)
# w = 1e-11
# std = torch.exp(w * log_var)
# eps = torch.randn_like(std)
# z = eps * std + mu
z = mu
print(z.shape)
recon = vae.decode(z)
recon = recon[0].view(1, 3, 150, 150)
# save_image(recon, f'face_output.jpg')
recon = recon.detach().numpy().squeeze()
recon = np.transpose(recon, (1, 2, 0))
plt.imshow(recon)
plt.show()
