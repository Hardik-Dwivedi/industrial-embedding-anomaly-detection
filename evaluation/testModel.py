import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # For memory-saving mixed precision
from ..training.hyperparameters import device, EMBEDDING_DIMS


def find_centroid(dataset, model):
    loader = DataLoader(dataset, shuffle= True, batch_size = 32)

    count = 0
    total = torch.zeros((1, EMBEDDING_DIMS)).to(device)

    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            logits, z = model(img)

            good_emb = z[label == 0]

            count += good_emb.shape[0]
            total += good_emb.sum(dim = 0, keepdim= True)

        centroid = total / count
        normalized_centroid = F.normalize(centroid, p = 2, dim = 1)

        del logits, z, total

        return normalized_centroid

def test_model(model, centroid, dataset):

    history = {
        'good_sample_distribution': [],
        'defective_sample_distribution': [],
        'ood_sample_distribution': []
        }

    model.eval()

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    centroid = centroid.to(device)

    sum_good = 0
    count_good = 0

    sum_def = 0
    count_def = 0

    sum_neg = 0
    count_neg = 0

    with torch.no_grad():

        for images, label in loader:

            images = images.to(device)
            label = label.to(device)

            logits, z = model(images)

            good_emb = z[label  == 0]
            def_emb = z[label == 1]
            neg_emb = z[label == 2]

            if def_emb.shape[0] > 0:
                dist = 1 - F.cosine_similarity(def_emb, centroid, dim=1)
                history['defective_sample_distribution'].append(dist)
                sum_def += dist.sum().item()
                count_def += def_emb.shape[0]

            if neg_emb.shape[0] > 0:
                dist = 1 - F.cosine_similarity(neg_emb, centroid, dim=1)
                history['ood_sample_distribution'].append(dist)
                sum_neg += dist.sum().item()
                count_neg += neg_emb.shape[0]
            
            if good_emb.shape[0] > 0:
                dist = 1 - F.cosine_similarity(good_emb, centroid, dim=1)
                history['good_sample_distribution'].append(dist)
                sum_good += dist.sum().item()
                count_good += good_emb.shape[0]

            del images, z, logits

    results = {}

    results['good_mean_distance'] = (
        sum_good / count_good if count_good > 0 else 0
    )

    results['defect_mean_distance'] = (
        sum_def / count_def if count_def > 0 else 0
    )

    results['ood_mean_distance'] = (
        sum_neg / count_neg if count_neg > 0 else 0
    )

    return results, history
