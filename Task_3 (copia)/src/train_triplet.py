#########################################################################################
#               BIOMETRICS & APPLIED INTELLIGENCE - 2nd Assignament - Task 3            # 
#                                     ~~~~~~~~~~~~~~~~                                  # 
#                   Authors: Javier Galan, Javier Muñoz, Pedro Delgado                  # 
#                                     ~~~~~~~~~~~~~~~~                                  #
# Comments:  run this file in order to perform ResNet34 + embedding architecture with   # 
# triplet loss function. Use areguments in order to set the paths to the path to save   # 
# the results, path to the model (backbone). Set batch-size (BS) learning rate (LR),    # 
# epochs (epochs) and train only with train or with train and validation (train_all     # 
# True). Remember to check that the correct paths are set in config file.               # 
# Time counter: 4 h                                                                     # 
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
from Triplet import modelEmbeddingT3
from dataset_utils import DeepFake, TripleTrainLoader, read_dataset, save_test_results
from train_utils import train_model, validate_model, test_model, initialize_weights_xavier_uniform, initialize_weights_xavier_normal, initialize_weights_he, computeROC, train_siamese_model
from config import TRAIN_SIAMESE_FILE, TRAIN_SIAMESE_ALL_FILE, VALIDATION_FILE, TEST_FILE, TRIPLET_CS, SEED, MEAN, STD, DATASET_PATH

def read_triplet(file_name):
# Reads .csv triplet format database.
# Input-> file_name: path to the csv file 
# outputs -> images paths and labels for anchors, positives samples and negatives samples of the dataset

    paths = []
    anchors_paths = []
    labels_anchors = []
    positives_paths = []
    labels_positives = []
    negatives_paths = []
    labels_negatives = []

    with open(file_name, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        for fila in csv_reader:
            anchors_paths.append(fila[0])
            labels_anchors.append(float(fila[1]))
            positives_paths.append(fila[2])
            labels_positives.append(float(fila[3]))
            negatives_paths.append(fila[4])
            labels_negatives.append(float(fila[5]))

    return anchors_paths, labels_anchors, positives_paths, labels_positives, negatives_paths, labels_negatives




if __name__ == "__main__":
   # Train Triplet embedding 
    parser = argparse.ArgumentParser(description="config file")
    parser.add_argument("--path", default="models/triplet/", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--BS", default=16, type=int)
    parser.add_argument("--LR", default=0.00005, type=float)
    parser.add_argument("--model_path", default="models/cnn/Experiment3/XU/Resnet34/Resnet34BS_16LR_0.0001E_25.pth", type=str)
    parser.add_argument("--show_model", default=False)
    parser.add_argument("--train_all", default=False)
    parser.add_argument("--create_example", default=False)
    args = parser.parse_args()

    RESULTS_PATH = args.path
    BATCH_SIZE = args.BS
    LEARNING_RATE = args.LR
    EPOCHS = args.epochs
 
    model_name = "siam_" + 'BS_' + str(BATCH_SIZE) + 'LR_' + str(LEARNING_RATE) + 'E_' + str(EPOCHS)
    # check for existence of the path where the results and the model are going to be saved
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

    # Set SEED in order to obtain deterministic results
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(SEED)  # CPU seed
    torch.cuda.manual_seed_all(SEED)  # GPU seed
    random.seed(SEED)  # python seed for image transformation
    np.random.seed(SEED)

   #Load the weights of the first step train and create the new model (3 embedding Layers)
    model = torchvision.models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 64),  
        nn.ReLU(inplace=True),
        nn.Linear(64, 1),  
        nn.Sigmoid())
    model.load_state_dict(torch.load(args.model_path))
    model_triplet = modelEmbeddingT3(model)

    if args.show_model == True:
        print(model_triplet)

   # Train with validation and train 
    if args.train_all == True:
        paths_anchors_T1, labels_anchores_T1, paths_positives_T1, labels_positives_T1, paths_negatives_T1, labels_negatives_T1 = read_triplet(os.path.join(DATASET_PATH, TRAIN_SIAMESE_ALL_FILE))        
        paths_anchors_T3, labels_anchores_T3, paths_positives_T3, labels_positives_T3, paths_negatives_T3, labels_negatives_T3 = read_triplet(os.path.join(DATASET_PATH, TRIPLET_CS))

        paths_anchors = paths_anchors_T1 + paths_anchors_T3
        labels_anchores = labels_anchores_T1 + labels_anchores_T3
        paths_positives = paths_positives_T1 + paths_positives_T3
        labels_positives = labels_positives_T1 + labels_positives_T3
        paths_negatives = paths_negatives_T1 + paths_negatives_T3
        labels_negatives = labels_negatives_T1 + labels_negatives_T3        

        triplet_dataset = TripleTrainLoader(paths_anchors, labels_anchores, paths_positives, labels_positives, paths_negatives ,labels_negatives, faceDetector, transform, model)
        triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True) 
   # Train only with train, no validation 
    else:
        paths_anchors, labels_anchores, paths_positives, labels_positives, paths_negatives, labels_negatives = read_triplet(os.path.join(DATASET_PATH, TRIPLET_CS))
        triplet_dataset = TripleTrainLoader(paths_anchors, labels_anchores, paths_positives, labels_positives, paths_negatives ,labels_negatives, faceDetector, transform, model)
        triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    
    # Select GPU if posible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using:{device}')

   # Adam optimizer and TripletLoss 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    tripletLoss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    
   #Train the model 
    loss_train = train_siamese_model(model_triplet, triplet_loader, optimizer, tripletLoss, EPOCHS, device, model_name, RESULTS_PATH)
   #Save the figure 
    x = range(1, len(loss_train)+1)
    plt.plot(x, loss_train, label='Train')
    plt.title('Training  Embedding')
    plt.ylabel('Triplet Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, 'train_embedding_loss.png'))
    plt.close()

   #Show an example of anchors, positive and negative samples 
    if args.create_example == True:

        a_i, _, p_i, _, n_i, _ = triplet_dataset.__getitem__(23)
        

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Convertir tensor a array NumPy y ajustar formato si es necesario
        anchor = a_i.permute(1, 2, 0).cpu().numpy()  # Permutar dimensiones y convertir a NumPy
        anchor = np.clip(anchor, 0, 1)  # Ajustar rango de valores a [0, 1] (si es necesario)
        positive = p_i.permute(1, 2, 0).cpu().numpy()  # Permutar dimensiones y convertir a NumPy
        positive = np.clip(positive, 0, 1)  # Ajustar rango de valores a [0, 1] (si es necesario)
        negative = n_i.permute(1, 2, 0).cpu().numpy()  # Permutar dimensiones y convertir a NumPy
        negative = np.clip(negative, 0, 1)  # Ajustar rango de valores a [0, 1] (si es necesario)

        axs[0].imshow(anchor)
        axs[0].set_title("Anchor")
        axs[0].axis('off')
        axs[1].imshow(positive)
        axs[1].set_title("Positive")
        axs[1].axis('off')
        axs[2].imshow(negative)
        axs[2].set_title("Negative")
        axs[2].axis('off')
        plt.show()
        plt.savefig(os.path.join(RESULTS_PATH, "ejemplo.jpg"))
