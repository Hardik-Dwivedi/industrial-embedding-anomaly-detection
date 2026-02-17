import torch

EMBEDDING_DIMS = 1024
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
GAMMA_P = 1
GAMMA_D = 0.8
GAMMA_N = 0.2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')