from helper_functions import get_unique_labels

DATASET = "plantvillage"
ROOT_DIR = r"dataset/plantvillage_dataset/"
LR = 0.0003
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4

# train loop
BATCH_SIZE = 64
TEST_SIZE = 0.1
TRAIN_SIZE = 1 - TEST_SIZE
EPOCHS = 15
USE_INDEX = True 
# callback
PATIENCE = 3

TOP_K_SAVES = 1
# training loop
NUM_TRIALS = 1

INDICES_DIR = "indices/"
CHECKPOINTS_DIR = "checkpoints/"
METRICS_DIR = "metrics/"
WANDB_PROJECT = "Plant_Disease_Classification"

# model directories
CONVNEXT_DIR = CHECKPOINTS_DIR + "convnext/"
AUTOENCODER_DIR = CHECKPOINTS_DIR + "autoencoder/"

# model file names
AUTOENCODER_FILENAME = "autoencoder_"
CONVNEXT_FILENAME = "convnext_"

# transformations
ORIGINAL_SIZE = (299, 299)
RESIZE = 236
CROP = 224
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]
ROTATION = 30

CLASS_NAMES = get_unique_labels(ROOT_DIR)
