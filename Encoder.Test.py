from TrainValSplit import test_dataset
from Encoder import SiameseEncoder
from CustomSiameseLoss import customSiameseLossFunction
import torch
from torch.utils.data import DataLoader
from Hyperparams import *
from torch.nn.functional import F

test_model = SiameseEncoder(EMBEDDING_DIMS).to(device)

checkpoint_path = '/kaggle/working/siamese_encoder_epoch_50.pth'
state_dict = torch.load(checkpoint_path, map_location=device)

test_model.load_state_dict(state_dict)

import gc
gc.collect()
torch.cuda.empty_cache()

loss_function = customSiameseLossFunction(GAMMA_P, GAMMA_D, GAMMA_N)

# Use a very small batch size for testing
safe_test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

test_model.eval()
val_loss, d_ap, d_an = 0.0, 0.0, 0.0

with torch.no_grad():
    for anc, pos, neg, neg_type in safe_test_loader:
        # Move to GPU
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        neg_type = neg_type.to(device).float()

        # Forward
        outputs = test_model(anc, pos, neg, neg_type)
        
        # Loss
        loss = loss_function(*outputs)
        val_loss += loss.item()

        # Distance
        d_ap += torch.mean(F.pairwise_distance(outputs[0], outputs[1])).item()
        d_an += torch.mean(F.pairwise_distance(outputs[0], outputs[2])).item()

        # MANUALLY CLEAR TENSORS to be safe
        del anc, pos, neg, outputs, loss
        
print(f"Finished successfully! Avg Loss: {val_loss/len(safe_test_loader):.4f}")
print(f"Avg d_ap: {d_ap/len(safe_test_loader)}")
print(f"Avg d_an: {d_an/len(safe_test_loader)}")