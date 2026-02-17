from torchvision.transforms import v2
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.transforms import InterpolationMode

weights = Wide_ResNet50_2_Weights.DEFAULT
preprocess = weights.transforms()

anchor_transform = preprocess

non_anchor_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5), # MVTec objects like 'screw' or 'grid' benefit from this
    v2.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
    preprocess # This handles the Resize, CenterCrop, and Normalization
])
