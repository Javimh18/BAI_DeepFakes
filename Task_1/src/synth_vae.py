from vae import VAE
from config import IM_HEIGHT, IM_WIDTH, MEAN, STD
from dataset_utils import VAE_DeepFake

import torch
from torch.utils.data.dataloader import default_collate
from torchsummary import summary
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from torchvision import transforms

if __name__ == '__main__':
    # device initialization
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "cpu"
        
    # parse arguments
    parser = argparse.ArgumentParser(description='Parsing VAE forwarding options.')
    # Add command-line arguments
    parser.add_argument('-i', '--input_image', help='Path to single image.', required=True)
    parser.add_argument('-o', '--output_image', help='Path of the output(generated) image.', required=True)
    parser.add_argument('-d', '--input_directory', help="Path to the directory with all the data.", required=False)
    parser.add_argument('-t', '--output_directory', help="Path to the directory where the generated data will be.", required=False)
    parser.add_argument('-p', '--path_to_weights', help="Path to the weights of the VAE to test.", required=True)
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.Resize((IM_HEIGHT,IM_WIDTH), antialias=True)
    ])
    
    # model definition and initialization
    vae = VAE(input_size=(3, IM_HEIGHT, IM_WIDTH),
                    latent_dim=2048,
                    n_convLayers=5,
                    n_convChannels=[256, 128, 64, 32, 64],
                    filter_sizes=[4, 3, 3, 3, 3],
                    strides=[2, 1, 1, 1, (2,1)], # change it for (1,2) in the last layer's stride
                    n_fcLayer=1, 
                    n_hidden_dims=[4096])
    
    vae.load_state_dict(torch.load(args.path_to_weights))
    vae.eval()
    vae.to(device)
        
    if args.input_image:
        pil_im = Image.open(args.input_image)
        tr_im = transform(pil_im)
        # forward pass
        tr_recon_im, mu, logvar, z = vae(tr_im.unsqueeze(0).to(device))
        
        # Denormalize the tensor
        denormalized_tensor = tr_recon_im.squeeze(0).clone()
        for channel in range(3):  # Assuming 3 channels
            denormalized_tensor[channel, :, :] = denormalized_tensor[channel, :, :] * STD[channel] + MEAN[channel]

        # Convert the PyTorch tensor to a PIL image
        denormalized_image = transforms.ToPILImage()(denormalized_tensor)
        denormalized_image.save(args.output_image)
        
        
        
        
         
    
    
    

