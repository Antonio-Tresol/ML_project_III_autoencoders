from helper_functions import get_unique_labels

DATASET = "plantvillage"
ROOT_DIR = r"dataset/plantvillage_dataset/"
LR = 0.001
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4

# train loop
BATCH_SIZE = 64
EPOCHS = 2
USE_INDEX = True
# callback
PATIENCE = 3
TOP_K_SAVES = 1
# training loop
NUM_TRIALS = 1

# mlp
MLP_HIDDEN_DIM = 64
MLP_HIDDEN_LAYERS = 3
MLP_INPUT_SIZE = 512

# train test
TRAIN_SIZE_80_20 = 0.8
TEST_SIZE_80_20 = 1-TRAIN_SIZE_80_20
TRAIN_SIZE_80_10_10 = (0.1 * 1)/TEST_SIZE_80_20
TEST_SIZE_80_10_10 = 1-TRAIN_SIZE_80_10_10

TRAIN_SIZE_50_50 = 0.5
TEST_SIZE_50_50 = 1-TRAIN_SIZE_50_50
TRAIN_SIZE_50_35_15 = (0.35 * 1)/TEST_SIZE_50_50
TEST_SIZE_50_35_15 = 1-TRAIN_SIZE_50_35_15

INDICES_DIR = "indices/"
CHECKPOINTS_DIR = "checkpoints/"
METRICS_DIR = "metrics/"
WANDB_PROJECT = "Plant_Disease_Classification"

# dataset
DATASET_80_20_NAME = DATASET + "_80_20"
DATASET_80_10_10_NAME = DATASET + "_80_10_10"
DATASET_50_50_NAME = DATASET + "_50_50"
DATASET_50_35_15_NAME = DATASET + "_50_35_15"

# model directories
CONVNEXT_80_20_DIR = CHECKPOINTS_DIR + "convnext_80_20/"
AUTOENCODER_80_20_DIR = CHECKPOINTS_DIR + "autoencoder_80_20/"
DENOISING_AUTOENCODER_80_20_DIR = CHECKPOINTS_DIR + "denoising_autoencoder_80_20/"
AUTOENCODER_CLASSIFIER_80_10_10_DIR = CHECKPOINTS_DIR + "autoencoder_classifier_80_10_10/"
FREEZED_AUTOENCODER_CLASSIFIER_80_10_10_DIR = CHECKPOINTS_DIR + "freezed_autoencoder_classifier_80_10_10/"
DENOISING_AUTOENCODER_CLASSIFIER_80_10_10_DIR = CHECKPOINTS_DIR + "denoising_autoencoder_classifier_80_10_10/"

CONVNEXT_50_50_DIR = CHECKPOINTS_DIR + "convnext_50_50/"
AUTOENCODER_50_50_DIR = CHECKPOINTS_DIR + "autoencoder_50_50/"
DENOISING_AUTOENCODER_50_50_DIR = CHECKPOINTS_DIR + "denoising_autoencoder_50_50/"
AUTOENCODER_CLASSIFIER_50_35_15_DIR = CHECKPOINTS_DIR + "autoencoder_classifier_50_35_15/"
FREEZED_AUTOENCODER_CLASSIFIER_50_35_15_DIR = CHECKPOINTS_DIR + "freezed_autoencoder_classifier_50_35_15/"
DENOISING_AUTOENCODER_CLASSIFIER_50_35_15_DIR = CHECKPOINTS_DIR + "denoising_autoencoder_classifier_50_35_15/"

# model file names
CONVNEXT_80_20_FILENAME = "convnext_80_20_"
AUTOENCODER_80_20_FILENAME = "autoencoder_80_20_"
DENOISING_AUTOENCODER_80_20_FILENAME = CHECKPOINTS_DIR + "denoising_autoencoder_80_20_"
AUTOENCODER_CLASSIFIER_80_10_10_FILENAME = "autoencoder_classifier_80_10_10_"
FREEZED_AUTOENCODER_CLASSIFIER_80_10_10_FILENAME = "freezed_autoencoder_classifier_80_10_10_"
DENOISING_AUTOENCODER_CLASSIFIER_80_10_10_FILENAME = "denoising_autoencoder_classifier_80_10_10_"

CONVNEXT_50_50_FILENAME = "convnext_50_50_"
AUTOENCODER_50_50_FILENAME = "autoencoder_50_50_"
DENOISING_AUTOENCODER_50_50_FILENAME = CHECKPOINTS_DIR + "denoising_autoencoder_50_50_"
AUTOENCODER_CLASSIFIER_50_35_15_FILENAME = "autoencoder_classifier_50_35_15_"
FREEZED_AUTOENCODER_CLASSIFIER_50_35_15_FILENAME = "freezed_autoencoder_classifier_50_35_15_"
DENOISING_AUTOENCODER_CLASSIFIER_50_35_15_FILENAME = "denoising_autoencoder_classifier_50_35_15_"

# transformations
ORIGINAL_SIZE = (299, 299)
RESIZE = 236
CROP = 224
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]
ROTATION = 30

CLASS_NAMES = get_unique_labels(ROOT_DIR)
