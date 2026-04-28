import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RoboticsDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # Lists all images and takes every 5th one to stay under 500
        all_files = sorted(os.listdir(image_dir))
        self.images = [f for f in all_files if f.endswith('.png') or f.endswith('.jpg')][::5][:500]
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), self.transform(img)