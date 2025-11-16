import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RAOrdinalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: dataset split folder (train/val/test)
        transform: torchvision transforms for augmentation & resizing
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # Expecting folder structure: root_dir/0/, root_dir/1/, ..., root_dir/4/
        for label in os.listdir(root_dir):
            label_folder = os.path.join(root_dir, label)

            if not os.path.isdir(label_folder):
                continue

            for img_name in os.listdir(label_folder):
                self.image_paths.append(os.path.join(label_folder, img_name))
                self.labels.append(int(label))  # convert folder name to integer label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
