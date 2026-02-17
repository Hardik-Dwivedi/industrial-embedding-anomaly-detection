import torch.nn as nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torch import nn
import torch.nn.functional as F

# Encoding neural network
class SiameseEncoder(nn.Module):

    def __init__(self, embedding_dim):
        super(SiameseEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.conv_backbone = nn.Sequential(*list(wide_resnet50_2(weights = Wide_ResNet50_2_Weights.DEFAULT).children())[:-1])
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU()
        )

    def forward(self, anchor, pos, neg, neg_type):

        y_anchor = self.flatten(self.conv_backbone(anchor))
        y_pos = self.flatten(self.conv_backbone(pos))
        y_neg = self.flatten(self.conv_backbone(neg))

        y_anchor = F.normalize(y_anchor, p=2, dim=1)
        y_pos = F.normalize(y_pos, p=2, dim=1)
        y_neg = F.normalize(y_neg, p=2, dim=1)

        return y_anchor, y_pos, y_neg, neg_type
