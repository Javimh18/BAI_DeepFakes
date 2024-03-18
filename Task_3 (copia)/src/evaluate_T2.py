from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import TASK2_PATH, TASK2_FILE, DATASET_PATH, SEED, MEAN, STD
from dataset_utils import DeepFake, read_dataset, save_test_results
from train_utils import train_model, validate_model, test_model, initialize_weights_xavier_uniform, initialize_weights_xavier_normal, initialize_weights_he, computeROC
from Triplet import modelEmbedding, modelReconstructed

import numpy as np
import torch.nn as nn

import os
import torch
import random
import argparse
import torchvision


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config file")
    parser.add_argument("--path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model", default="Resnet34", type=str)
    args = parser.parse_args()

    # Set SEED in order to obtain deterministic results
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(SEED)  # CPU seed
    torch.cuda.manual_seed_all(SEED)  # GPU seed
    random.seed(SEED)  # python seed for image transformation
    np.random.seed(SEED)

    if args.model == "Resnet34":
        print(f"Using {args.model}")
        model = torchvision.models.resnet34()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 64),  
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  
            nn.Sigmoid() 
        )
        model.load_state_dict(torch.load(args.model_path))
    elif args.model == "Triplet":   
        original_model = torchvision.models.resnet34()
        original_model.fc = nn.Sequential(
            nn.Linear(original_model.fc.in_features, 64),  
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  
            nn.Sigmoid()
        )
        model_embedding = modelEmbedding(original_model)
        for param in model_embedding.parameters():
            param.requires_grad = False
        model = modelReconstructed(model_embedding)
        model.load_state_dict(torch.load(args.model_path))

    faceDetector = MTCNN(keep_all=True)

    eval_transform = transforms.Compose([ 
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
     ])

    paths_list, images_list, labels_list = read_dataset(os.path.join(DATASET_PATH, TASK2_FILE))
    test_dataset = DeepFake(paths_list, images_list, labels_list, eval_transform, faceDetector)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Select GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using:{device}')

    criterion = nn.BCELoss()

    loss_test, accuracy_test, list_test_outputs, list_test_labels = test_model(model, test_loader, criterion, device)
    auc = computeROC(list_test_labels, list_test_outputs, os.path.join(args.path, "ROC.jpg"))
    print(f"EVALUATION OVER TASK 2 DATASET=> loss:{loss_test}, accuracy:{accuracy_test}, AUC:{auc[0]}")

    