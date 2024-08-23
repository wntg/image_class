
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils.datasets import MyDataset
from utils.utils import set_seed
from models.xception import model_selection
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import sys
from models.swin import Swin,EfficientNet

from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
parser = argparse.ArgumentParser(description = 'test')
parser.add_argument("--batch_size", type=int,default=128)
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
parser.add_argument('--exp_dir', type=str,default='outputs/xception_noaug')
    
def main():
    epoch = 10
    args, overrides = parser.parse_known_args(sys.argv[1:])

    save_path = args.exp_dir
    os.makedirs(save_path, exist_ok=True)
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')
    
    set_seed(args.seed)

  
    batch = args.batch_size
  
    train_dataset = MyDataset('dataset/phase1/trainset_label.txt',"train")
    #train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=False,num_workers=64)
    val_dataset = MyDataset('dataset/phase1/valset_label.txt',"val")
    #val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False,num_workers=64)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch,  sampler=train_sampler,num_workers=32)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch*2,sampler=val_sampler,num_workers=32)

    device = "cuda" if torch.cuda.is_available else "cpu"


    model = model_selection('xception',num_out_classes=2).to(device)
    #model = Swin().to(device)
    #model = EfficientNet().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    model = DDP(model, device_ids=[rank], output_device=rank,find_unused_parameters=True)

    loss = nn.CrossEntropyLoss()
    model_optim = optim.Adam(model.parameters(), lr=0.0001)


    train_loss, val_acc = train(rank,model, epoch, save_path, model_optim, loss, device, train_loader,train_sampler, val_loader)
    if rank == 0:
        plot(save_path,train_loss, val_acc)

def train_epoch(scaler,rank,epoch, model, model_optim, loss, train_loader, device):

    train_loss = 0.0
    model.train()
    train_loss = []
    if rank==0:
        process = tqdm(train_loader, desc='train epoch %d ' % epoch, position=0)

    for data in train_loader:
        img, label = data[0].to(device), data[1].to(device)
        model_optim.zero_grad()
        # 在autocast enable 区域运行forward
        with autocast():
            output = model(img)
            losses = loss(output, label)
        # losses.backward()
        scaler.scale(losses).backward()
        # scaler 更新参数，会先自动unscale梯度
        # 如果有nan或inf，自动跳过
        scaler.step(model_optim)
        # scaler factor更新
        scaler.update()
        # model_optim.step()
        train_loss.append(losses.item())
        if rank == 0:
            process.set_description(desc='train epoch %d ' % epoch + 'loss :%0.5f' % (np.average(train_loss)))
            process.update(1)
    if rank == 0:
        process.close()  # 关闭进度条
    return np.average(train_loss)

def val_epoch(rank,epoch, model, loss, val_loader, device):
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    val_loss = []
    if rank == 0:
        process = tqdm(val_loader, desc='val epoch %d ' % epoch, position=0)
        
    with torch.no_grad():
        for data in val_loader:
            img, label = data[0].to(device), data[1].to(device)
            output = model(img)
            losses = loss(output, label)
            val_loss.append(losses.item())

            _, preds = torch.max(output, 1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)
            if rank == 0:
                process.set_description('val epoch %d ' % epoch + 'loss :%0.5f , acc:%0.2f' % (np.average(val_loss), 100 * val_correct / val_total) )
                process.update(1)
        val_total = torch.tensor(val_total).cuda()
        val_correct = torch.tensor(val_correct).cuda()
        dist.reduce(val_correct,dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(val_total,dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            val_accuracy = 100 * val_correct / val_total
            process.close()  # 关闭进度条
            print(f'Validation Accuracy: {val_accuracy:.2f}%')

            return val_accuracy
        else:
            return None
    
def train(rank,model, epoch, save_path, model_optim, loss, device, train_loader,train_sampler, val_loader):
    train_losses = []
    val_acc = []
    best_acc = 0.0
    # GradScaler对象用来自动做梯度缩放 
    scaler = GradScaler()
    for epoch in range(epoch):
        train_sampler.set_epoch(epoch) #保证每个进程采样过程随机数种子一样
        train_loss = train_epoch(scaler,rank,epoch, model, model_optim, loss, train_loader, device)
        val_accuracy = val_epoch(rank,epoch, model, loss, val_loader, device)
        if rank == 0:
            if val_accuracy > best_acc:
                torch.save(model, f'{save_path}/best.pth')
                print('Saving model...')
                best_acc = val_accuracy
            
            train_losses.append(train_loss)
            val_acc.append(val_accuracy.detach().cpu().numpy())
        dist.barrier()

    return train_losses, val_acc

def plot(output,train_loss, val_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training loss')
    plt.savefig(output+'/train_loss.png')
    plt.show()
        
    plt.figure(figsize=(10, 5))
    plt.plot(val_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation accuracy')
    plt.savefig(output+'/val_acc.png')
    plt.show()
    



if __name__ == '__main__':
    main()

