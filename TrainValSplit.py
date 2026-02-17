from sklearn.model_selection import train_test_split
from Dataset import MVTec_Dataset
from torch.utils.data import Subset

full_dataset = MVTec_Dataset(root_dir='../input/datasets/ipythonx/mvtec-ad')

indices = list(range(len(full_dataset)))
labels = [full_dataset.anchor_map[i][0] for i in indices]

# Splitting the indices
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Generating the train and test subsets
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, val_idx)

print(f'Total Anchors: {len(indices)}')
print(f'Total Train Anchors: {len(train_dataset)}')
print(f'Total Validation Anchors: {len(test_dataset)}')
