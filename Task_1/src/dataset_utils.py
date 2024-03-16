import csv, cv2, torch, os
from config import DATASET_PATH
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def read_dataset(file_name):
    # Function to read the .csv training files
    # Input: file_name -> .csv path 
    # Outputs: lists with the content of the .csv
    #       paths -> path of the image in the dataset 
    #       images -> images names
    #       labels -> corresponding labels

    paths = []
    images = []
    labels = []

    with open(file_name, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        next(csv_reader)
        for fila in csv_reader:
            paths.append(fila[0])
            images.append(fila[1])
            labels.append(fila[2])

    return paths, images, labels

def save_test_results(list_outputs, list_labels, file_name):

    # Function to save the test outputs and true labels in order to test diferent thresholds
    # Input:
    #       list_outputs -> predicted outputs for the test dataset (obtained from function test_model)
    #       list_labels -> true labels of the test dataset  (obtained from function test_model)
    #       file_name -> path to save the resulting .csv

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Outputs', 'Labels'] )
        for item1, item2 in zip(list_outputs, list_labels):
            writer.writerow([item1, item2])

class DeepFake(Dataset):
    def __init__(self, paths, images, labels, transform, facedetector):
        self.paths = paths
        self.images = images
        self.labels = labels
        self.transform = transform
        self.facedetector = facedetector

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        margin = 25

        path_image = os.path.join(DATASET_PATH, self.paths[item], self.images[item])
        image = Image.open(path_image)
       # print(np.shape(image))
        boxes, _ = self.facedetector.detect(image)
       # print(boxes)

        if np.any(boxes):
            x1 = int(boxes[0][0])
            y1 = int(boxes[0][1])
            x2 = int(boxes[0][2])
            y2 = int(boxes[0][3])
            
            image = image.crop((x1-margin, y1-margin, x2+margin, y2+margin))

        image_transform = self.transform(image)
        label = self._get_label(item)
        label_transform = torch.tensor(float(label))

        return image_transform, label_transform

    def _get_label(self, item):
        return self.labels[item]

    
    
class VAE_DeepFake(Dataset):
    def __init__(self, dataset_root, subset, transform_pre=None, transform=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.subset = subset 
        self.transform_pre = transform_pre
        self.transform = transform
        
        # load different splits from the original dataset
        files_path = os.path.join(dataset_root, self.subset)
        
        self.path_to_ims = []
        for cur_dir, sub_dir, files in os.walk(files_path):
            if not sub_dir and files: # only get the files where there is no subdirectory
                for f_n in files:
                    self.path_to_ims.append(os.path.join(cur_dir, f_n))
                    
    def __len__(self):
        return(len(self.path_to_ims))
    
    def __getitem__(self, idx):
        path_im = self.path_to_ims[idx]
        im = Image.open(path_im)
        
        # Tranformations for the image to be loaded
        if self.transform_pre:
            im = self.transform_pre(im)
        if self.transform:
            im = self.transform(im)
        
        label = -1
        if 'fake' == path_im.split('/')[2]:
            label = 1
        elif 'real' == path_im.split('/')[2]:
            label = 0
        # Control scenario, if there is somthing wrong with the image's path
        if label == -1:
            print(f"Paths are not loaded propertly: {path_im}")
            exit(-1)
            
        label = torch.tensor(float(label))
        
        return idx, path_im, (im, label)
