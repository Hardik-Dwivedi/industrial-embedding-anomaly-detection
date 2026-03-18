import random
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ProdCheck.dataset.transforms import ContrastiveTransform, transforms

class Recog_Dataset(Dataset):
    def __init__(self, root_dir, transforms=None, mode = 'train', split_ratio = 0.8):
        super().__init__()
    
        self.target_category = 'metal_nut'
        self.transforms = ContrastiveTransform() if mode == 'train' else transforms['operational_transform']
        self.root_path = Path(root_dir)
        self.mode = mode

        # 1. We secure all the good images for the recognizer
        img_paths = []
        for obj_type in self.root_path.iterdir():
            if obj_type.is_dir():
                good_dir =  obj_type / 'train' / 'good'

                if good_dir.exists():
                    img_paths.extend(good_dir.glob('*.png'))
        
        split_idx = int(len(self.img_paths) * split_ratio)
        random.seed(42)
        random.shuffle(img_paths)

        if mode is 'train':
            self.img_paths = img_paths[:split_idx]
        else:
            self.img_paths = img_paths[split_idx:]

   
    def __len__(self):
        # We define an epoch as 2000 random triplets to ensure stable gradient descent
        return len(self.img_paths)
    
    def __getitem__(self, index):

        # 1. Stochastic Anchor/Positive Selection
        in_path = self.img_paths[index]
        
        # Load and convert
        in_img = Image.open(in_path).convert('RGB')
        
        if self.mode == 'test':
            i1_tensor, i2_tensor = self.transforms(in_img)
            return i1_tensor, i2_tensor
        else:
            i1_tensor = self.transforms(in_img)

            return i1_tensor