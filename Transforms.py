from torchvision.transforms import v2
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

weights = Wide_ResNet50_2_Weights.DEFAULT
preprocess = weights.transforms()

anchor_transform = preprocess
non_anchor_transforms = v2.Compose([
    v2.RandomHorizontalFlip(0.5),
    v2.RandomRotation(10),
    preprocess
])
