import numpy as np

MODEL_TYPE = 'ae'

if (MODEL_TYPE == 'vae'):
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
    
elif (MODEL_TYPE == 'ae'):
    from keras import models
    from attention import AttentionLayer
    from ssim import SSIMLoss, SSIM_MSE_Loss

    # Load the model
    autoencoder = models.load_model(
        'models/03-0.1987.keras',
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
    
else:
    print(f'Error. Invalid model type: {MODEL_TYPE}')