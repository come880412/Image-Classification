import numpy as np
from Dataset import Cls_data
from torch.utils.data import DataLoader
import torch.nn as nn
from model import resnext101_32x8d
import torch
import sys

def test(model ,test_loader, csv_path):
    cuda = True if torch.cuda.is_available() else False
    model.eval()
    if cuda:
        model = model.cuda()
    csv_save = [['image_id', 'label']]
    for image, image_name in test_loader:
        with torch.no_grad():
            if cuda:
                image = image.cuda()
            pred = model(image)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            for i in range(len(image_name)):
                csv_save.append([image_name[i], pred_label[i]])
    np.savetxt(csv_path, csv_save, encoding='utf-8-sig', fmt='%s', delimiter=',')

if __name__ == '__main__':
    test_data_path = sys.argv[1]
    csv_path = sys.argv[2]
    saved_model = 'model_epoch9_acc89.60.pth?dl=1'

    test_data = Cls_data(test_data_path, 'test')
    test_loader = DataLoader(test_data, batch_size=64,shuffle=False, num_workers=0)
    model = resnext101_32x8d(pretrained=True)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, 50)
    # print(f'loading pretrained model from {saved_model}')
    model.load_state_dict(torch.load(saved_model))
    test(model, test_loader, csv_path)