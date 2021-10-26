from sklearn.manifold import TSNE
import csv
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Cls_data
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch
import tqdm
import random

def feature_extract():
    feature_list = []
    label_list = []
    saved_model = '../../saved_models/p1/resnext101_32x8d/model_epoch9_acc89.60.pth'
    val_path = '../../hw1_data/p1_data/val_50'
    val_data = Cls_data(val_path, 'val')
    val_loader = DataLoader(val_data, batch_size=256,shuffle=False, num_workers=0)
    model = models.resnext101_32x8d(pretrained=True)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, 50)
    model.load_state_dict(torch.load(saved_model))
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model = model.cuda()
    for image, label, image_name in tqdm.tqdm(val_loader):
        with torch.no_grad():
            image = image.cuda()
            pred = model(image).squeeze()
            pred = pred.cpu().detach().numpy()
            for i in range(len(pred)):
                feature_list.append(pred[i])
                label_list.append(label[i])
    return feature_list, label_list

if __name__ == '__main__':
    color_dict = mcolors.CSS4_COLORS
    # color_list = []
    # for key in color_dict.keys():
    #     color_list.append(key)
    color_list = ['peru', 'dodgerblue', 'brown', 'darkslategray', 'lightsalmon', 'orange', 'aquamarine', 'springgreen', 'chartreuse', 'fuchsia',
	      'mediumspringgreen', 'burlywood', 'palegreen', 'orangered', 'lightcoral', 'tomato', 'pink', 'darkseagreen', 'olive', 'darkgoldenrod',
              'turquoise', 'plum', 'darkmagenta', 'deeppink', 'red', 'slategrey', 'darkviolet', 'darkturquoise', 'skyblue', 'mediumorchid',
	      'magenta', 'deepskyblue', 'darkorchid', 'teal', 'wheat', 'green', 'lightcyan', 'royalblue', 'sienna', 'seagreen', 
	      'blueviolet', 'darkorange', 'aqua', 'purple', 'darkred', 'salmon', 'orchid', 'lightgreen', 'cadetblue', 'thistle']
    # random_num_list = np.random.choice(len(color_list), 50, replace=False).astype(int)
    # color_list = color_list[random_num_list]
    print(color_list)
    X, y = feature_extract()
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    tsne = TSNE(n_components=2, init='random', random_state=5, verbose=1)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(32, 32))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=color_dict[color_list[y[i]]], 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./t-sne.png')
    plt.show()