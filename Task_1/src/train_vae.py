from vae import VAE
from config import IM_HEIGHT, IM_WIDTH, DATASET_PATH, MEAN, STD, \
    VAE_ALPHA, VAE_EPOCHS, VAE_LR, VAE_WEIGHT_DECAY, VAE_BETAS, VAE_REG_PAR
from dataset_utils import VAE_DeepFake
from train_utils import VAETrainer

import torch
from torch.utils.data.dataloader import default_collate
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adamax
from torchvision import transforms

import os
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':
    # device initialization
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "cpu"
    
    # model definition and initialization
    model = VAE(input_size=(3, IM_HEIGHT, IM_WIDTH),
                    latent_dim=512,
                    n_convLayers=5,
                    n_convChannels=[512, 256, 128, 64, 32],
                    filter_sizes=[4, 3, 3, 3, 3],
                    strides=[2, 2, 1, 1, (2,1)], # change it for (1,2) in the last layer's stride
                    n_fcLayer=1, 
                    n_hidden_dims=[1024])
    summary(model.to(device), (3, IM_HEIGHT, IM_WIDTH))
    
    transform_pre_train = transforms.Compose([
        # Rotación con una probabilidad del 50%
        transforms.RandomRotation(degrees=20, interpolation=Image.BILINEAR),
        # Flip horizontal o vertical con una probabilidad del 50%
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    transform_pre_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_dataset = VAE_DeepFake(DATASET_PATH, 'train', transform_pre_train, \
        transforms.Resize((IM_HEIGHT,IM_WIDTH), antialias=True))
    val_dataset = VAE_DeepFake(DATASET_PATH, 'validation', transform_pre_val, \
        transforms.Resize((IM_HEIGHT,IM_WIDTH), antialias=True))
    
        # instanciate dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=32, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count(),
                                  collate_fn=default_collate)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=32, 
                                shuffle=True, 
                                num_workers=os.cpu_count(),
                                collate_fn=default_collate)
    
    # Setting up for training
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adamax(trainables, lr=VAE_LR)
    trainer = VAETrainer(model, 
                         optimizer, 
                         train_dataloader, 
                         val_dataloader, 
                         VAE_EPOCHS, 
                         VAE_REG_PAR)
    train_data, val_data = trainer.train()
    
    # obtaining data for loss
    train_loss = []
    train_mse_loss = []
    train_kl_loss = []
    val_loss = []
    val_mse_loss = []
    val_kl_loss = []
    for tr_d, val_d in zip(train_data, val_data):
        train_loss.append(tr_d['loss'])
        train_mse_loss.append(tr_d['loss_recon'])
        train_kl_loss.append(tr_d['loss_kl'])
        
        val_loss.append(val_d['val_loss'])
        val_mse_loss.append(val_d['val_loss_recon'])
        val_kl_loss.append(val_d['val_loss_kl'])
       
    save_dir = './results/vae/figures'
    print(f"INFO: Saving results under {save_dir} directory...")
    # plotting results for total loss 
    plt.figure(figsize=(10,6))
    plt.plot(train_loss, label='Train total loss')
    plt.plot(val_loss, label='Val total loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Total Loss evolution over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.jpg'))
    
    # plotting results for MSE loss 
    plt.figure(figsize=(10,6))
    plt.plot(train_mse_loss, label='Train reconstruction loss')
    plt.plot(val_mse_loss, label='Val reconstruction loss')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss evolution over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_recons.jpg'))
    
    # plotting results for KL loss 
    plt.figure(figsize=(10,6))
    plt.plot(train_kl_loss, label='Train KL loss')
    plt.plot(val_kl_loss, label='Val KL loss')
    plt.xlabel('Epochs')
    plt.ylabel('KL Loss')
    plt.title('KL Loss evolution over epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_kl.jpg'))
    
    