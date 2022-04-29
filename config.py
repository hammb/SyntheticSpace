import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FOLD = 0
CHECKPOINTS = "/home/AD/b556m/projects/tutorials/SyntheticSpace"
TASK = "Task003_mprage2space"
TRAIN_DIR = "/home/AD/b556m/data/SyntheticSpace/preprocessed_data/tasks/" + TASK

LEARNING_RATE = 2e-4
BATCH_SIZE = 1
RAND_SAMPLE_SIZE = 2
NUM_WORKERS = 10
IMAGE_SIZE = 256
CHANNELS_IMG = 1
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = True
LOAD_MODEL_PATH = None
RESUME_TRAINING = False
RESUME_TRAINING_PATH = None
RESUME_TRAINING_EPOCH = 0
PRE_TRAINING = False
SAVE_MODEL = True
PATCH_SIZE = (256, 256, 244)

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

CHECKPOINT_DISC_BEST = "disc_best.pth.tar"
CHECKPOINT_GEN_BEST = "gen_best.pth.tar"

MAX_VALUE_SPACE = 1582
MAX_VALUE_MPRAGE = 987