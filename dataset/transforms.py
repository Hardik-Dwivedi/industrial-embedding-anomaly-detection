from torchvision.transforms import v2
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.transforms import InterpolationMode
import torch
from PIL import Image

from torchvision.transforms import v2
import torch
class ContrastiveTransform:
    def __init__(self, size=224, mode='train'):
        self.mode = mode

        if mode == 'train':

            self.augment = v2.Compose([
                v2.Resize((size, size)),

                # small rotations are fine
                v2.RandomRotation(20),

                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),

                v2.RandomApply([
                    v2.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.1,
                        hue=0.05
                    )
                ], p=0.5),

                v2.RandomApply([
                    v2.GaussianBlur(kernel_size=3)
                ], p=0.2),

                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.augment = v2.Compose([
                v2.Resize((size, size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __call__(self, x):
        return self.augment(x)

