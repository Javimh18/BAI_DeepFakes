#########################################################################################
#               BIOMETRICS & APPLIED INTELLIGENCE - 2nd Assignament - Task 3            # 
#                                     ~~~~~~~~~~~~~~~~                                  # 
#                   Authors: Javier Galan, Javier Muñoz, Pedro Delgado                  # 
#                                     ~~~~~~~~~~~~~~~~                                  #
# Comments:                                                                             # 
# Time counter:                                                                         # 
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
import argparse
import random
import os

from facenet_pytorch import MTCNN, InceptionResnetV1

from dataset_utils import DeepFake, read_dataset, save_test_results
from train_utils import train_model, validate_model, test_model, initialize_weights_xavier_uniform, initialize_weights_xavier_normal, initialize_weights_he, computeROC
from config import TRAIN_FILE, VALIDATION_FILE, TRAIN_CS, VALIDATION_CS, TEST_FILE, TASK2_FILE, SEED, MEAN, STD, DATASET_PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="config file")
    parser.add_argument("--path", default="models", type=str)
    parser.add_argument("--model", default="Resnet50", type=str)
    parser.add_argument("--LR", default=0.0001, type=float)
    parser.add_argument("--BS", default=16, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--initialization", default=None, type=str)
    parser.add_argument("--data_aug", default=None, type=str)
    parser.add_argument("--show_model", default=False)
    parser.add_argument("--train_all", default=False)
    args = parser.parse_args()

    RESULTS_PATH = args.path
    MODEL = args.model
    BATCH_SIZE = args.BS
    LEARNING_RATE = args.LR
    EPOCHS = args.epochs
    INIT = args.initialization
    AUG = args.data_aug

    if args.train_all == True:
        BESTS = False
    else:
        BESTS = True

    # Model name (in order to save results and weights)
    model_name = MODEL + 'BS_' + str(BATCH_SIZE) + 'LR_' + str(LEARNING_RATE) + 'E_' + str(EPOCHS)

    # check for existence of the path where the results and the model are going to be saved
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Initialize facedetector 
    faceDetector = MTCNN(keep_all=True)

    # Default transformation: resize to 256, 256, convert to tensor and normalization for evaluation 
    eval_transform = transforms.Compose([ 
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
     ])

    if AUG == None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    elif AUG == "erasing":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    elif AUG == "vflip":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    elif AUG == "hflip":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    elif  AUG == "all":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])


    # Create the train, validation adn test dataloaders 

    if args.train_all == True:
        paths_train, images_train, labels_train = read_dataset(os.path.join(DATASET_PATH, TRAIN_FILE))
        paths_val,  images_val, labels_val = read_dataset(ps.path.join(DATASET_PATH, VALIDATION_FILE))
        paths_train_cs, images_train_cs, label_val_cs = read_dataset(os.path.join(DATASET_PATH, TRAIN_CS_FILE))


        paths_list = paths_val + paths_train + paths_train_cs
        images_list = images_val + images_train + images_train_cs
        labels_list = labels_val + labels_train + label_val_cs


        train_dataset = DeepFake(paths_list, images_list, labels_list, transform, faceDetector)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, VALIDATION_CS))
        validation_dataset = DeepFake(paths_list, images_list, labels_list, eval_transform, faceDetector)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    
    else:
        paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, TRAIN_CS))
        train_dataset = DeepFake(paths_list, images_list, labels_list, transform, faceDetector)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, VALIDATION_CS))
        validation_dataset = DeepFake(paths_list, images_list, labels_list, eval_transform, faceDetector)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, TASK2_FILE))
    test_dataset = DeepFake(paths_list, images_list, labels_list, eval_transform, faceDetector)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    if MODEL == "Resnet50":
        print(f"Using {MODEL}")
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 64),  
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  
            nn.Sigmoid()  
        ) 
    
    if args.show_model == True:
        print(model)

    # Set SEED in order to obtain deterministic results
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(SEED)  # CPU seed
    torch.cuda.manual_seed_all(SEED)  # GPU seed
    random.seed(SEED)  # python seed for image transformation
    np.random.seed(SEED)

    # Select GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using:{device}')

    # Optimizer Adam and Binary Cross Entropy Loss 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Perform training 
    list_loss_train, list_acc_train, list_loss_val, list_acc_val = train_model(model, train_loader, validation_loader, \
                                                                               test_loader, optimizer, criterion, EPOCHS, device, RESULTS_PATH, best=BESTS)
    # Best model over the validation dataset  
    best_model = model
    best_model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_in_val.pth')))
    # Obtain best validation results for printing 
    best_val_acc, best_val_loss, best_val_ROC = validate_model(best_model, validation_loader, criterion, device, RESULTS_PATH)
    # Perform Test  
    loss_test, accuracy_test, list_test_outputs, list_test_labels = test_model(best_model, test_loader, criterion, device)
    
    # Save the loss and accuracy figure  
    x = range(1, len(list_loss_train)+1)

    plt.plot(x, list_loss_train, label='Train')
    plt.plot(x, list_loss_val, label='Validation')
    plt.title('Training and Validation Loss Function')
    plt.ylabel('BCELoss()')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'train_loss.png'))
    plt.close()

    plt.plot(x, list_acc_train, label='Train')
    plt.plot(x, list_acc_val, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('[%]')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'train_acc.png'))
    plt.close()

    # Save the model weigths 
    torch.save(model.state_dict(), os.path.join(RESULTS_PATH, model_name + '.pth'))

    # Save test results for threshold testing 
    save_test_results(list_test_outputs, list_test_labels, os.path.join(RESULTS_PATH, model_name + '_test_results.csv'))

    # Compute ROC and save the figure 
    roc_auc, fpr, tpr = computeROC(list_test_labels, list_test_outputs, RESULTS_PATH)

    print(f'Test => Loss: {loss_test:.4f}, Accuracy:{accuracy_test:.4f}%, AUC:{roc_auc:.4f}')

    with open(os.path.join(RESULTS_PATH, "results.txt"), 'w') as file:
        file.write(f"Validation => acc: {best_val_acc} loss: {best_val_loss} AUC:{best_val_ROC}\n")
        file.write(f"Test acc: {accuracy_test} loss: {loss_test} AUC:{roc_auc}\n")
    