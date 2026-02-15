import random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from Transforms import anchor_transform, non_anchor_transforms

class MVTec_Dataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        # Save Transforms
        self.transforms = transforms or {
            'anchor': anchor_transform, 
            'non-anchor': non_anchor_transforms
        }
        
        self.root_path = Path(root_dir)
        self.data = dict()
        self.anchor_map = []
        
        # Categories
        self.object_types = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
                            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
                            'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

        for obj in self.object_types:
            obj_test_path = self.root_path / obj / 'test'
            
            # 1. Load Anchors 
            good_paths = [str(p) for p in (obj_test_path / 'good').glob('*.png')]
            
            # 2. Load Defectives 
            defective_data = {}
            if obj_test_path.exists():
                for defect_dir in obj_test_path.iterdir():
                    if defect_dir.is_dir() and defect_dir.name != 'good':
                        paths = [str(p) for p in defect_dir.glob('*.png')]
                        if paths: defective_data[defect_dir.name] = paths

            self.data[obj] = {'anchors': good_paths, 'defectives': defective_data}

            # Map global index to (object_name, image_index)
            for i in range(len(good_paths)):
                self.anchor_map.append((obj, i))

    def __len__(self):
        return len(self.anchor_map)
    
    def __getitem__(self, index):
        obj_type, idx = self.anchor_map[index]
        obj_data = self.data[obj_type]

        # 1. Retrieve anchor
        anchor_img = Image.open(obj_data['anchors'][idx]).convert('RGB')

        # 2. Retrieve positive image
        pos_path = random.choice(obj_data['anchors'])
        pos_img = Image.open(pos_path).convert('RGB')

        # 3. Retrieve neg image
        if random.random() < 0.5 and obj_data['defectives']:    # Get defective item of same class as neg
            # Same object, different condition (Negative Type 0)
            def_cat = random.choice(list(obj_data['defectives'].keys()))
            neg_path = random.choice(obj_data['defectives'][def_cat])
            neg_type = 0 
        else:
            # Entirely different object (Negative Type 1 - Fraud detection)
            other_obj = random.choice([t for t in self.object_types if t != obj_type])
            neg_path = random.choice(self.data[other_obj]['anchors'])
            neg_type = 1
        
        neg_img = Image.open(neg_path).convert('RGB')
        
        # Apply the specific transforms
        anchor_tensor = self.transforms['anchor'](anchor_img)
        pos_tensor = self.transforms['non-anchor'](pos_img)
        neg_tensor = self.transforms['non-anchor'](neg_img)
        
        return anchor_tensor, pos_tensor, neg_tensor, neg_type