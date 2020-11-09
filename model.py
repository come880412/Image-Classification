import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import math
import torch.utils.model_zoo as model_zoo

class VGG16_fine_tune(nn.Module):
    def __init__(self,features,num_classes=50):
        super(VGG16_fine_tune, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
                nn.Linear(512*7*7,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096,num_classes),    
        )
        for p in self.features.parameters():
            p.requires_grad = False
    
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x1 = x
        for i in range(len(self.classifier)):
            x1 = self.classifier[i](x1)
            if i ==3:
                x_embed = x1
        x = self.classifier(x)
        return x,x_embed

