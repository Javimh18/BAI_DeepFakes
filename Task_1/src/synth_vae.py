from vae import VAE
from config import VAE_IM_HEIGHT, VAE_IM_WIDTH, MEAN, STD

import torch
from torchsummary import summary
from torchvision import transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm


def save_image_and_latent_space(vae: torch.nn.Module, input_im: str, output_im: str, feat_path: str, transform: transforms.Compose):
    """
    This function loads an input image, performs forward pass through the VAE model to reconstruct the image 
        and obtain its corresponding latent space. It then denormalizes the reconstructed image, converts it to 
        a PIL image, and saves it. Additionally, it saves the latent space features as a numpy array.

    Args:
        vae (torch.nn.Module): The trained variational autoencoder model.
        input_im (str): The path to the input image file.
        output_im (str): The path to save the reconstructed image.
        feat_path (str): The path to save the latent space features.
        transform (transforms.Compose): A torchvision.transforms.Compose object containing image transformations.

    Returns:
        None: The function does not return any value
    """
    pil_im = Image.open(input_im)
    tr_im = transform(pil_im)
    # forward pass of the variational autoencoder, it returns the reconstructed image and the latent space (z)
    tr_recon_im, _, _, z = vae(tr_im.unsqueeze(0).to(device))

    # Denormalize the tensor
    denormalized_tensor = tr_recon_im.squeeze(0).clone()
    for channel in range(3):  # Assuming 3 channels
        denormalized_tensor[channel, :, :] = denormalized_tensor[channel, :, :] * STD[channel] + MEAN[channel]

    # Convert the PyTorch tensor to a PIL image and save it.
    denormalized_image = transforms.ToPILImage()(denormalized_tensor)
    denormalized_image.save(output_im)
    
    # Convert tensor to numpy array and save it.
    np.save(feat_path, z.detach().cpu().numpy())


if __name__ == '__main__':
    # device initialization
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "cpu"
        
    # parse arguments
    parser = argparse.ArgumentParser(description='Parsing VAE forwarding options.')
    # Add command-line arguments
    parser.add_argument('-i', '--input_image', help='Path to single image.', required=False)
    parser.add_argument('-o', '--output_image', help='Path of the output(generated) image.', required=False)
    parser.add_argument('-d', '--input_dir', help="Path to the directory with all the data.", required=False)
    parser.add_argument('-t', '--output_dir', help="Path to the directory where the generated data will be.", required=False)
    parser.add_argument('-f', '--features_dir', help="Path to the features directory where the latent space of the image(s) is going to be stored", required=True)
    parser.add_argument('-p', '--path_to_weights', help="Path to the weights of the VAE to test.", required=True)
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # little checks on parameter consistency
    if args.input_image and not args.output_image:
        print("ERROR: Input image specified <-i, --input_image> but output image not specified <-o, --output_image>.\n \
            Please specify both of them.\n \
            EXITING..")
        exit(-1)
    elif (args.input_image and args.output_image) and (args.input_dir or args.output_dir):
        print("ERROR: Input and output image specified <-i, --input_image> & <-o, --output_image>, \
            but also input and output directories <-d, --input_directory> & <-t, --output_directory>.\n \
            Please specify either and input and output image or directory, but not both.\n \
            EXITING...")
        exit(-1)
    
    # create directories if there is not any
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.Resize((VAE_IM_HEIGHT,VAE_IM_WIDTH), antialias=True)
    ])

    # model definition and initialization
    vae = VAE(input_size=(3, VAE_IM_HEIGHT, VAE_IM_WIDTH),
                    latent_dim=2048,
                    n_convLayers=5,
                    n_convChannels=[256, 128, 64, 32, 16],
                    filter_sizes=[4, 3, 3, 3, 3],
                    strides=[2, 1, 1, 1, (2,1)], # change it for (1,2) in the last layer's stride
                    n_fcLayer=1, 
                    n_hidden_dims=[4096])
    
    vae.load_state_dict(torch.load(args.path_to_weights))
    vae.eval()
    vae.to(device)
    print("Model Weights loaded. File processing starting...")
    
    # saving image information
    if args.input_image:
        save_image_and_latent_space(vae, args.output_image, args.feat_dir, transform)
    else:
        for im_name in tqdm(os.listdir(args.input_dir)):
            if 'fake' in args.input_dir:
                im = im_name.split(".")[1]
            else:
                im = im_name.split(".")[0]
            im_path = os.path.join(args.input_dir, im_name)
            out_path = os.path.join(args.output_dir, im_name)
            feat_path = os.path.join(args.features_dir, im+'.npy')
            save_image_and_latent_space(vae, im_path, out_path, feat_path, transform)
        
    print("Done.")    
        
        
        
        
         
    
    
    

