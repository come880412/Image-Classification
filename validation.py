import numpy as np
import argparse
from Dataset import Cls_data
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from model import resnext101_32x8d
from torch.optim.lr_scheduler import LambdaLR
import torch
import tqdm
import sys
from torchsummary import summary

def validation(model ,val_loader, csv_path):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
    if cuda:
        model = model.cuda()
    acc = 0.
    correct_total = 0
    label_total = 0
    csv_save = [['image_id', 'label']]
    for image, label, image_name in tqdm.tqdm(val_loader):
        with torch.no_grad():
            if cuda:
                image = image.cuda()
                label = label.cuda()
            pred = model(image)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100
            for i in range(len(image_name)):
                csv_save.append([image_name[i], pred_label[i]])
    acc = (correct_total / label_total) * 100
    print('validation accuracy:', round(acc, 3), '%')
    np.savetxt(csv_path, csv_save, encoding='utf-8-sig', fmt='%s', delimiter=',')


if __name__ == '__main__':
    val_data_path = sys.argv[1]
    csv_path = sys.argv[2]
    saved_model = 'model_epoch9_acc89.60.pth?dl=1'

    val_data = Cls_data(val_data_path, 'val')
    val_loader = DataLoader(val_data, batch_size=64,shuffle=False, num_workers=0)
    model = resnext101_32x8d(pretrained=True)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, 50)
    print(f'loading pretrained model from {saved_model}')
    model.load_state_dict(torch.load(saved_model))
    summary(model, (3, 224, 224))
    validation(model, val_loader, csv_path)