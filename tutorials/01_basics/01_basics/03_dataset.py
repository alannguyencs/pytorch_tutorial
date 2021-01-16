from torch.utils.data import Dataset, DataLoader

"""
Dataset:
1. Load data (image, text, number...), load labels
2. Preprocessing: rescale image, encode data
3. Augmentation: rotate, flip, crop, gaussian noise...
4. Move current data type (e.g. from numpy) to torch tensors

Dataloader:
Split dataset into batches
multi-processing
shuffle in case of training data
"""

class CustomDataset(Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0


# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = DataLoader(dataset=custom_dataset,
                           batch_size=64,
                           shuffle=True)