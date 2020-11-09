#torch
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np

import os
from Dataset import Cls_data
from model import VGG16_fine_tune
import util
import csv

def test(model):
    val_root = '../hw2_data/p1_data/val_50'
    valset = Cls_data(val_root,mode = 'test')
    valset_loader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=1)
    val_loss = 0
    correct = 0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data,target in valset_loader:
            data, target = data.to(device), target.to(device)
            output,emb = model(data)
            val_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(valset_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(valset_loader.dataset),
        100. * correct / len(valset_loader.dataset)))

def extract_feature(model):
    val_root = '../hw2_data/p1_data/val_50'
    valset = Cls_data(val_root,mode = 'test')
    valset_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=1)
    model.eval()
    plt.figure(figsize=(16, 16))
    with open('feature.csv','w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    with torch.no_grad():
        for data,target in valset_loader:
            data, target = data.to(device), target.to(device)
            output,emb = model(data)
            emb = np.array(emb.cpu())
            y = np.array(target.cpu())
            with open('feature.csv','a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                wrt = np.concatenate((emb,y[:,np.newaxis]),axis=1)
                for i in range(wrt.shape[0]):
                    writer.writerow(wrt.tolist()[i])
            

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(0)

    VGG16 = models.vgg16(pretrained=True)
    model = VGG16_fine_tune(VGG16.features).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum =0.9)
    util.load_checkpoint('./model/epoch40(acc_75.280000).pth',model,optimizer)
    extract_feature(model)
    # test(model)
