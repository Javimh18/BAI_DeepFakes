#########################################################################################
#               BIOMETRICS & APPLIED INTELLIGENCE - 2nd Assignament - Task 1            # 
#                                     ~~~~~~~~~~~~~~~~                                  # 
#                   Authors: Javier Galan, Javier Muñoz, Pedro Delgado                  # 
#                                     ~~~~~~~~~~~~~~~~                                  #
# Comments:                                                                             # 
# Time counter: 6 h                                                                     # 
#########################################################################################

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_curve, auc

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import torchvision
import torch.optim
import random
import os

from dataset_utils import DeepFake, read_dataset, save_test_results
from train_utils import train_model, test_model, initialize_weights_xavier_uniform, initialize_weights_xavier_normal
from config import TRAIN_FILE, VALIDATION_FILE, TEST_FILE, DATASET_PATH, CNN_WEIGHT_INIT,\
    CNN_MODEL_NAME, CNN_BATCH_SIZE, CNN_LEARNING_RATE, CNN_NUM_EPOCHS, SEED, MEAN, STD


## TO DO: ##
# Early stopping
# Otras arquitecturas???
# Data augmentation

if __name__ == '__main__':
    
    # Model name (in order to save results and weights)
    model_name = CNN_MODEL_NAME + 'BS_' + str(CNN_BATCH_SIZE) + 'LR_' + str(CNN_LEARNING_RATE) + 'E_' + str(CNN_NUM_EPOCHS)
    # Path to save model weights
    save_model_path = os.path.join('models/cnn/', model_name)  
    # Path to save training figures, ROC, test results
    save_model_results = os.path.join('results/cnn/', model_name) 
    
    # check for existence of the path where the results and the model are going to be saved
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        
    if not os.path.exists(save_model_results):
        os.makedirs(save_model_results)
        
    # Default transformation: resize to 512, 512, convert to tensor and normalization  
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Create the train, validation adn test dataloaders 
    paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, TRAIN_FILE))
    train_dataset = DeepFake(paths_list, images_list, labels_list, transform)
    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE)

    paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, VALIDATION_FILE))
    validation_dataset = DeepFake(paths_list, images_list, labels_list, transform)
    validation_loader = DataLoader(validation_dataset, batch_size=CNN_BATCH_SIZE)

    paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, TEST_FILE))
    test_dataset = DeepFake(paths_list, images_list, labels_list, transform)
    test_loader = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE)

    # Set SEED in order to obtain deterministic results
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(SEED)  # CPU seed
    torch.cuda.manual_seed_all(SEED)  # GPU seed
    random.seed(SEED)  # python seed for image transformation
    np.random.seed(SEED)

    # Load ResNet18 model and add an extra linear layer to fit the binary classification problem  
    model = torchvision.models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),  
        nn.ReLU(inplace=True),
        nn.Linear(512, 1),  
        nn.Sigmoid()  
    )

    # Select the weight initialization technique (in case is seleted) 
    if CNN_WEIGHT_INIT == 'Xavier_Uniform':
        print(f'Initialization with Xavier Uniform...')
        model.apply(initialize_weights_xavier_uniform)
    elif CNN_WEIGHT_INIT == 'Xavier_Normal':
        print(f'Initialization with Xavier Normal...')
        model.apply(initialize_weights_xavier_normal)
    elif CNN_WEIGHT_INIT == 'He':
        print(f'Initialization with He (Kaiming Uniform)...')
        model.apply()
    else:
        print(f'No initialization of weights')
        pass

    print(model)

    # Select GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using:{device}')

    # Optimizer Adam and Binary Cross Entropy Loss 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CNN_LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Perform training 
    list_loss_train, list_acc_train, list_loss_val, list_acc_val = train_model(model, train_loader, validation_loader, \
                                                                               test_loader, optimizer, criterion, CNN_NUM_EPOCHS, device)
    # Test model  
    list_loss_test, list_acc_test, list_test_outputs, list_test_labels = test_model(model, test_loader, criterion, device)
    
    # Save the loss and accuracy figure  
    x = range(1, len(list_loss_train)+1)

    plt.plot(x, list_loss_train, label='Train')
    plt.plot(x, list_loss_val, label='Validation')
    plt.title('Training and Validation Loss Function')
    plt.ylabel('BCELoss()')
    plt.legend()
    plt.savefig(os.path.join(save_model_results, 'train_loss.png'))
    plt.close()

    plt.plot(x, list_acc_train, label='Train')
    plt.plot(x, list_acc_val, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('[%]')
    plt.legend()
    plt.savefig(os.path.join(save_model_results, 'train_acc.png'))
    plt.close()

    # Save the model weigths 
    torch.save(model.state_dict(), os.path.join(save_model_path, model_name + '.pth'))

    # Save test results for threshold testing 
    save_test_results(list_test_outputs, list_test_labels, os.path.join(save_model_results, model_name + '_test_results.csv'))

    # Compute ROC and save the figure 
    fpr, tpr, thresholds = roc_curve(list_test_labels, list_test_outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_model_results, 'ROC.png'))
    plt.close()    
    
    debug = 1