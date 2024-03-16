# General purpose constants
DATASET_PATH = 'dataset'
TRAIN_FILE = 'train_dataset.csv'
VALIDATION_FILE = 'validation_dataset.csv'
TEST_FILE = 'test_dataset.csv'

TASK2_PATH = "dataset/evaluation/"
TASK2_FILE = "evaluation_dataset.csv"

IM_HEIGHT = 512
IM_WIDTH = 512 
MEAN = [0.326, 0.283, 0.313]
STD = [0.219, 0.179, 0.192]
SEED = 1234


# VAE loss weight parameters
VAE_BATCH_SIZE = 64
VAE_ALPHA = 5e4
VAE_EPOCHS = 500
VAE_LR = 5e-2 # lr = 0.01
VAE_WEIGHT_DECAY = 5e-7
VAE_BETAS = (0.95, 0.999)
VAE_REG_PAR = 0# 5e-3 at first, but after few epochs should be increasing to give the KL Loss more importance
VAE_IM_HEIGHT = 64
VAE_IM_WIDTH = 64