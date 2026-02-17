import torch
import torch.optim as optim
from CustomSiameseLoss import CustomSiameseLossFunction
from Encoder import SiameseEncoder
from TrainValSplit import train_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Hyperparams import *

# Create encoder model
model = SiameseEncoder(EMBEDDING_DIMS).to(device)

# Define loss function and the optimizer
loss_function = CustomSiameseLossFunction(GAMMA_P,GAMMA_D, GAMMA_N)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Genrate Dataset and DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle= True, num_workers=2, pin_memory=True)

#Begin training

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    total_d_ap = 0.0
    total_d_an = 0.0
    
    for i, (anc, pos, neg, neg_type) in enumerate(train_loader):

        # Move all tensors to device (gpu)
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        neg_type = neg_type.to(device)


        # Setting grads to zero
        optimizer.zero_grad()

        # Obtain model output
        outputs = model(anc, pos, neg, neg_type)

        # Finding loss
        loss = loss_function(*outputs)

        # Run backprop
        loss.backward()
        optimizer.step()

        # Accumulating loss
        running_loss += loss.item()
        
        # Accumulate the distance metric
        with torch.no_grad():
            total_d_ap += torch.mean(F.pairwise_distance(outputs[0], outputs[1])).item()
            total_d_an += torch.mean(F.pairwise_distance(outputs[0], outputs[2])).item()

    # Calculating Epoch Averages
    avg_loss = running_loss / len(train_loader)
    avg_d_ap = total_d_ap / len(train_loader)
    avg_d_an = total_d_an / len(train_loader)

    # Summarizing for epoch
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f} | Avg D_ap: {avg_d_ap:.3f} | Avg D_an: {avg_d_an:.3f}")

    # Saving every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'siamese_encoder_epoch_{epoch+1}.pth')

