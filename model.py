from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torch import nn
base_model = wide_resnet50_2(weights = Wide_ResNet50_2_Weights)

layers = [base_model.children()]

feature_extractor = nn.Sequential(*layers)

