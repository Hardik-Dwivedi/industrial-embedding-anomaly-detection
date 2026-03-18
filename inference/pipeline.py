from ..dataset.transforms import ContrastiveTransform
from ..models.encoder import SiameseEncoder_v1
from ..training.hyperparameters import EMBEDDING_DIMS
import torch
import numpy as np


def infer(image):
    model = SiameseEncoder_v1(EMBEDDING_DIMS)
    state_dict = torch.load('../data/checkpoint.pt')
    model.load_state_dict(state_dict)

    transform = ContrastiveTransform(224, 'eval')

    img_tensor = transform(image)

    logits, z = model(img_tensor)

    del img, img_tensor, logits, z
    return np.argmax(logits)
