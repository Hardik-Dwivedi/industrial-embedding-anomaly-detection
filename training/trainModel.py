import torch
import torch.optim as optim
from ..models.encoder import SiameseEncoder_v1
from ..dataset.dataset import MVTecTestDataset
from ..dataset.transforms import ContrastiveTransform
from torch.utils.data import DataLoader
import torch.nn.functional as F
from hyperparameters import device, data_directory, BATCH_SIZE
import torch.nn as nn
from ProdCheck.training.earlyStopping import EarlyStopping

from torch.amp import autocast, GradScaler # For memory-saving mixed precision

def train_model_v1(embedding_dims, num_epochs, learning_rate, early_stop=True, ):
    # 1. Clear any leftover memory from previous crashes
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # 2. Initialize Model, Optimizer, and Scaler
    model = SiameseEncoder_v1(embedding_dim=embedding_dims).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate)
    
    # GradScaler is essential for Mixed Precision (Half-precision math)
    scaler = GradScaler() 

    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    # 3. DataLoader - IMPORTANT: Start with BATCH_SIZE = 8 or 16
    loader = DataLoader(
        MVTecTestDataset(data_directory, ContrastiveTransform(224, mode= 'train')),
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        corr , incorr = 0
        count = 0
        for i, (img, label) in enumerate(loader):

            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad(set_to_none=True) # set_to_none=True saves more memory

            # 4. Forward pass with Mixed Precision (Autocast)
            with autocast(device_type='cuda'):
                # Your model already does F.normalize internally
                logits, z = model(img)    # logits has shape (B, 3)
                loss = loss_function(logits, label)

            # 5. Backward pass with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            mask = torch.argmax(logits, dim=1)
            
            corr += (mask == label).sum().item()
            incorr += (mask != label).sum().item()

            count += len(label)
            # 6. Immediate Cleanup of batch tensors
            del img, logits, z, loss
        
        # End of Epoch Stats
        acc = corr / count

        print(f'Epoch [{epoch + 1}/{num_epochs}] | Accuracy: {acc:.4f}')

        # 7. Early Stopping & Cache Reset
        if early_stop:
            early_stopping(acc, model)
            if early_stopping.early_stop:
                print('Early Stopping triggered. Training halted!')
                break
        
        # Optional: Clean up cache after every epoch to prevent fragmentation
        torch.cuda.empty_cache()

    return model