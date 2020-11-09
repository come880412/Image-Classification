#torch
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy
import os
from Dataset import Cls_data
from model import VGG16_fine_tune
from Resnet import resnet18
from Densenet import densenet121
import util

#use gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.cuda.set_device(0)

def train(model, epoch, log_interval,train_root,val_root):
    trainset = Cls_data(train_root,mode = 'train')
    trainset_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum =0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    iteration = 0
    print('Start to train')
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output,emb = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
        
            iteration += 1
        val(model,ep,optimizer,val_root)

def val(model,ep,optimizer,val_root):
    valset = Cls_data(val_root,mode = 'val')
    valset_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=1)
    val_loss = 0
    correct = 0
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data,target in valset_loader:
            data, target = data.cuda(), target.cuda()
            output,emb = model(data)
            val_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    util.save_checkpoint('./model/res18SGD/epoch%d(acc:%0f).pth'%(ep,100. * correct / len(valset_loader.dataset)),model,optimizer)
    val_loss /= len(valset_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(valset_loader.dataset),
        100. * correct / len(valset_loader.dataset)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 50, help='How many epochs you want to run', type=int)
    parser.add_argument('--gpu_id', default = 0, help='Choose gpu_id', type=int)
    parser.add_argument('--interval', default = 20, type=int)
    parser.add_argument('--train_root', default = '../hw2_data/p1_data/train_50', type=str)
    parser.add_argument('--val_root', default = '../hw2_data/p1_data/val_50', type=str)
    parser.add_argument('--num_classes', default =50, type=int)
    args = parser.parse_args()


    #load VGG16 model
    # VGG16 = models.vgg16(pretrained=True)
    # model = VGG16_fine_tune(VGG16.features).to(device)
    # VGG16_state = VGG16.state_dict()
    # model_state = model.state_dict()
    # states_to_load = {}
    # for name, param in VGG16_state.items():
    #     if name.startswith('feature'):
    #         states_to_load[name] = param
    # model_state.update(states_to_load)
    # model.load_state_dict(model_state)
    # train(model, args.epochs, args.interval,args.train_root,args.val_root)

    #load resnet18
    Resnet18 = resnet18(pretrained=True).cuda()
    print(Resnet18)
    # for param in resnet18.parameters():
    #     param.requires_grad = False
    # resnet18.fc = nn.Linear(512, args.num_classes).cuda()
    # train(resnet18,args.epochs,args.interval,args.train_root,args.val_root)
    #load resnet50
    # resnet50 = resnet50(pretrained = True).cuda()
    # for param in resnet50.parameters():
    #     param.requires_grad = False
    # resnet50.fc = nn.Linear(2048, args.num_classes).cuda()
    # train(resnet50,args.epochs,args.interval,args.train_root,args.val_root)

    #Densenet
    # Densenet121 = densenet121(pretrained = True).cuda()
    # for param in Densenet121.parameters():
    #     param.requires_grad = False
    # Densenet121.classifier  = nn.Linear(1024, args.num_classes).cuda()
    # train(Densenet121,args.epochs,args.interval,args.train_root,args.val_root)
    # for name, param in Densenet121.named_parameters(): #檢查是否有把其他的grad關掉
    #     if param.requires_grad:
    #         print(name, param.data)
    


    

    
    