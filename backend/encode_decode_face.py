import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
from vae import LATENT_DIM, IMAGE_SIZE

# Load the model
vae = torch.load('models/vae_model_20.pth', map_location='cpu')

def encode(image):
    image = image.astype('float32') / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions to [1, 3, H, W]
    mu, _ = vae.encode(image)
    z = mu
    z = z.detach().numpy()
    return z

def decode(latent):
    latent = latent.astype('float32')
    latent = torch.from_numpy(latent)
    latent = latent.unsqueeze(0)
    # print(f'latent shape before decode: {latent.shape}')
    recon = vae.decode(latent)
    recon = recon[0].view(1, 3, 150, 150)
    recon = recon.detach().numpy().squeeze() * 255.0
    recon = np.transpose(recon, (1, 2, 0))
    return recon

def get_latent_length():
    return LATENT_DIM

def get_latent_shape():
    return (1, LATENT_DIM)