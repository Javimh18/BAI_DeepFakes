#########################################################################################
#               BIOMETRICS & APPLIED INTELLIGENCE - 2nd Assignament - Task 1 & 2        # 
#                                     ~~~~~~~~~~~~~~~~                                  # 
#                   Authors: Javier Galan, Javier Muñoz, Pedro Delgado                  # 
#                                     ~~~~~~~~~~~~~~~~                                  #
# Comments: run this file in order to perform fine tunning  over a ResNet34 +           # 
# embedding architecture. Use areguments in order to set the paths to the path to save  # 
# the results, paths to all the modeles (backbone and embedding). Set batch-size (BS)   # 
# learning rate (LR), epochs (epochs) and train only with train or with train and       # 
# validation (train_all True). Remember to check that the correct paths are set in      # 
# config file.                                                                          # 
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
import argparse
import random
import csv
import os

from facenet_pytorch import MTCNN, InceptionResnetV1
from Triplet import modelEmbeddingT3, modelReconstructedT3
from dataset_utils import DeepFake, TripleTrainLoader, read_dataset, save_test_results
from train_utils import train_model, validate_model, test_model, initialize_weights_xavier_uniform, initialize_weights_xavier_normal, initialize_weights_he, computeROC, train_siamese_model
from config import TRAIN_SIAMESE_FILE, TRAIN_FILE, TASK2_FILE, VALIDATION_FILE, TEST_FILE, TRAIN_CS, VALIDATION_CS, SEED, MEAN, STD, DATASET_PATH



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="config file")
    parser.add_argument("--path", default="models/finne_tune", type=str)
    parser.add_argument("--epochs", default = 10, type=int)
    parser.add_argument("--BS", default=4, type=int)
    parser.add_argument("--LR", default=0.00005, type=float)
    parser.add_argument("--embedding_path", default="models/siam/siam_BS_16LR_0.0001E_25.pth", type=str)
    parser.add_argument("--original_path", default="models/cnn/Experiment3/XU/Resnet34/Resnet34BS_16LR_0.0001E_25.pth", type=str)
    parser.add_argument("--train_all", default=False)
    args = parser.parse_args()

    RESULTS_PATH = args.path
    BATCH_SIZE = args.BS
    LEARNING_RATE = args.LR
    EPOCHS = args.epochs

    model_name = "Fine_tunned" + 'BS_' + str(BATCH_SIZE) + 'LR_' + str(LEARNING_RATE) + 'E_' + str(EPOCHS)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Initialize facedetector 
    faceDetector = MTCNN(keep_all=True)

    # Default transformation: resize to 256, 256, convert to tensor and normalization for evaluation 
    transform = transforms.Compose([ 
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    eval_transform = transforms.Compose([ 
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Set SEED in order to obtain deterministic results
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(SEED)  # CPU seed
    torch.cuda.manual_seed_all(SEED)  # GPU seed
    random.seed(SEED)  # python seed for image transformation
    np.random.seed(SEED)

   # Load the pretrained weights 
    original_model = torchvision.models.resnet50()
    original_model.fc = nn.Sequential(
        nn.Linear(original_model.fc.in_features, 64),  
        nn.ReLU(inplace=True),
        nn.Linear(64, 1),  
        nn.Sigmoid()
    )
    original_model.load_state_dict(torch.load(args.original_path))
   
   # Create the embedding model 
    model_embedding = modelEmbeddingT3(original_model)
    model_embedding.load_state_dict(torch.load(args.embedding_path))
    print(model_embedding)
   # Freeze the weights 
    for param in model_embedding.parameters():
        param.requires_grad = False
   # Create the new model  
    model_reconstructed = modelReconstructedT3(model_embedding)
   # print(model_reconstructed)

   # Read the selected dataset 
    if args.train_all == True:
        paths_train, images_train, labels_train = read_dataset(os.path.join(DATASET_PATH, TRAIN_FILE))
        paths_val,  images_val, labels_val = read_dataset(ps.path.join(DATASET_PATH, VALIDATION_FILE))
        paths_train_cs, images_train_cs, label_val_cs = read_dataset(os.path.join(DATASET_PATH, TRAIN_CS))


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using:{device}')

    optimizer = torch.optim.Adam(params=model_reconstructed.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    list_loss_train, list_acc_train, list_loss_val, list_acc_val = train_model(model_reconstructed, train_loader, validation_loader, \
                                                                               test_loader, optimizer, criterion, EPOCHS, device, RESULTS_PATH, best=True)
    best_model = model_reconstructed
    best_model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_in_val.pth')))
    # Obtain best validation results for printing 
    best_val_acc, best_val_loss, best_val_ROC = validate_model(best_model, validation_loader, criterion, device, RESULTS_PATH)
    # Perform Test  
    loss_test, accuracy_test, list_test_outputs, list_test_labels = test_model(best_model, test_loader, criterion, device)
    
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

    save_test_results(list_test_outputs, list_test_labels, os.path.join(RESULTS_PATH, model_name + '_test_results.csv'))

    # Compute ROC and save the figure 
    roc_auc, fpr, tpr = computeROC(list_test_labels, list_test_outputs, RESULTS_PATH)

    print(f'Test => Loss: {loss_test:.4f}, Accuracy:{accuracy_test:.4f}%, AUC:{roc_auc:.4f}')

    with open(os.path.join(RESULTS_PATH, "results.txt"), 'w') as file:
        file.write(f"Validation => acc: {best_val_acc} loss: {best_val_loss} AUC:{best_val_ROC}\n")
        file.write(f"Test acc: {accuracy_test} loss: {loss_test} AUC:{roc_auc}\n")
    