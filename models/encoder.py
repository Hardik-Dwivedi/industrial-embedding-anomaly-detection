import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import torch.nn.functional as F

# Encoding neural network
class SiameseEncoder_v1(nn.Module):

    def __init__(self, embedding_dim):
        super(SiameseEncoder_v1, self).__init__()

        self.embedding_dim = embedding_dim
        self.conv_backbone = nn.Sequential(*list(wide_resnet50_2(weights = Wide_ResNet50_2_Weights.DEFAULT).children())[:-1])
        for params in self.conv_backbone.parameters():
            params.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.embedding_dim)
        )
        self.classifier = nn.Linear(self.embedding_dim, 3)


    def forward(self, img):

        x = self.conv_backbone(img)
        z = self.embedding(x)
        z = F.normalize(z, p = 2, dim = 1)

        logits = self.classifier(z)

        return logits, z