import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class MyDataset(Dataset):
    def __init__(self, txt, mode):
        self.img_path = []
        self.label = []
        self.mode = "dataset/phase1/trainset" if mode == "train" else "dataset/phase1/valset"
        #self.mode = "dataset/tiger/train" if mode == "train" else "dataset/tiger/train"
        
        with open(txt, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip().split(',')[0] == 'img_name':
                    continue
                
                self.img_path.append(line.strip().split(',')[0])
                self.label.append(line.strip().split(',')[1])
        
        if mode == "train":
            self.transform = transforms.Compose([
                # transforms.RandomAffine(degrees=0, scale=(0.6, 1)),
                # transforms.ColorJitter(brightness=0.4, contrast=0, saturation=0.2, hue=0),
                # transforms.RandomHorizontalFlip(p=0.5),

                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        # 测试
        # self.img_path = self.img_path[:1280]
        # self.label = self.label[:1280]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        image_path = os.path.join(self.mode, self.img_path[i])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(int(self.label[i]), dtype=torch.long)
        
        return image, label
    