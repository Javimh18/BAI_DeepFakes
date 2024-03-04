# General purpose constants
DATASET_PATH = 'dataset'
TRAIN_FILE = 'train_dataset_shuffled.csv'
VALIDATION_FILE = 'validation_dataset.csv'
TEST_FILE = 'test_dataset.csv'
IM_HEIGHT = 256
IM_WIDTH = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 1234

# CNN Specific constants
CNN_BATCH_SIZE = 16
CNN_NUM_EPOCHS = 5
CNN_LEARNING_RATE = 0.001
CNN_MODEL_NAME = 'ResNet18' # Name of the model 
CNN_WEIGHT_INIT = 'Xavier_Normal' # Init -> Xavier_Normal, Xavier_Uniform (tested), He

# VAE loss weight parameters
VAE_ALPHA = 5e4
VAE_EPOCHS = 500
VAE_LR = 1e-4 # lr = 0.01
VAE_WEIGHT_DECAY = 5e-7
VAE_BETAS = (0.95, 0.999)
VAE_REG_PAR = 1e-5 # reg_par = 1e-5
