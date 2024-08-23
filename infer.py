import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm
import os
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torchvision
import argparse
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.xception import model_selection
import torch.distributed as dist
from PIL import Image
from models.swin import Swin
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345', rank=0, world_size=1)

class MyDataset(Dataset):
    def __init__(self, csv, mode):
        self.img_path = pd.read_csv(csv)['img_name']
        self.mode = "/mnt/nanwu/project/DeepfakeBench/preprocessing/dataset_json/phase2/testset1_seen" 
        
        if mode == "train":
            self.transform = transforms.Compose([
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

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, i):
        image_path = os.path.join(self.mode, self.img_path[i])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return self.img_path[i], image
    
batch = 16 #512
dataset = MyDataset('/mnt/nanwu/project/DeepfakeBench/preprocessing/dataset_json/phase2/testset1_seen_nolabel.txt',"infer")
dataloader = DataLoader(dataset, batch_size=batch, shuffle=False,num_workers=32)

device = "cuda" if torch.cuda.is_available else "cpu"


exp_dir='outputs/xception_noaug'
model = torch.load(exp_dir+"/best.pth")
model.eval()

results = []
process = tqdm(dataloader, desc='infer')
with torch.no_grad():
    for data in process:
        img_name, img = data[0], data[1].to(device)
        output = model(img)
        prob = F.softmax(output, dim=1)
        prob = prob.cpu().numpy()
        for i in range(len(img_name)):
            results.append([img_name[i], prob[i][1].item()])

df = pd.DataFrame(results, columns=['img_name', 'y_pred'])
df.to_csv(exp_dir+'/'+exp_dir.split('/')[-1]+'_submission.csv', index=False)