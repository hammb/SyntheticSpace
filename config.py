import torch
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FOLD = 0
CHECKPOINTS = os.environ['ss_checkpoint_path']
TASK = ""
TRAIN_DIR = os.path.join(os.environ['ss_data_path'], TASK)

LEARNING_RATE = 2e-4
BATCH_SIZE = 1
RAND_SAMPLE_SIZE = 2
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 1
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 400
LOAD_MODEL = False
LOAD_MODEL_PATH = None
RESUME_TRAINING = False
RESUME_TRAINING_PATH = None
RESUME_TRAINING_EPOCH = 0
PRE_TRAINING = False
SAVE_MODEL = True
PATCH_SIZE = (256, 256, 244)
AUGMENTATION = True
TRAINER = ""

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

CHECKPOINT_DISC_BEST = "disc_best.pth.tar"
CHECKPOINT_GEN_BEST = "gen_best.pth.tar"

MAX_VALUE_SPACE = 1582
MAX_VALUE_MPRAGE = 987