from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)
transforms_val =    transforms.Compose([
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rgb_std),])

transforms_test =    transforms.Compose([
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rgb_std),])

transforms_train =  transforms.Compose([
                    # transforms.Resize(224),
                    transforms.RandomRotation(5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rgb_std),
                    transforms.RandomErasing(),])
class Cls_data(Dataset):
    def __init__(self,root,mode):
        self.root = root
        if mode == 'train':
            self.transform = transforms_train
        elif mode == 'val':
            self.transform = transforms_val
        else:
            self.transform = transforms_test
        self.filename = []

        #Load data
        for i in os.listdir(self.root):
            filename = os.path.join(self.root,i)
            label = int(i.split('_')[0])
            self.filename.append([filename,label])


    def __getitem__(self,index):
        image_root,label = self.filename[index]
        image = Image.open(image_root)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.filename)