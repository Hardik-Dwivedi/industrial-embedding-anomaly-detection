import torch
from torch.utils.data import DataLoader
from ..training.hyperparameters import device
from torch.nn import functional as F

def collect_embeddings(model, dataset, batch_size=32):

    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            logits, z = model(images)

            z = F.normalize(z, dim=1)

            all_embeddings.append(z.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings.numpy(), labels.numpy()