import os
import sys
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.net import SPPNet
from dataset.cityscapes import CityscapesDataset
from utils.preprocess import minmax_normalize


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
model = SPPNet(output_channels=19).to(device)
model_path = '../model/cityscapes_deeplab_v3_plus/model.pth'
param = torch.load(model_path)
model.load_state_dict(param)
del param

batch_size = 1

valid_dataset = CityscapesDataset(split='test', net_type='deeplab')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# images_list = []
# labels_list = []
# preds_list = []

model.eval()
with torch.no_grad():
    for i, batched in enumerate(valid_loader):
        image, label, image_path = batched['img'], batched['lbl'], batched['img_path']

        image_name, _ = os.path.splitext(image_path[0])
        image_np = image.numpy().transpose(0, 2, 3, 1)
        label_np = label.numpy()

        image, label = image.to(device), label.to(device)
        pred = model.tta(image, net_type='deeplab')
        pred = pred.argmax(dim=1)
        pred_np = pred.detach().cpu().numpy()

        # Ignore index
        pred_np[label_np == 255] = 0
        # label_np[label_np == 255] = 0
        
        # image_np = minmax_normalize(image_np, norm_range=(0, 1), orig_range=(-1, 1)) * 255.
        # label = Image.fromarray(np.squeeze(label_np).astype(np.uint8))
        pred = Image.fromarray(np.squeeze(pred_np).astype(np.uint8))
        # image = Image.fromarray(np.squeeze(image_np).astype(np.uint8))

        # label.save(os.path.join('./results/', str(i)+'_label.png'))
        print('{}/{}: Processing {}...'.format(i+1, len(valid_loader), os.path.join(image_name)))
        pred.save(os.path.join(image_name+'_pred_labels.png'))
        # image.save(os.path.join('./results/', str(i)+'_image.png'))

        # images_list.append(images_np)
        # labels_list.append(labels_np)
        # preds_list.append(preds_np)

        # if len(images_list) == 4:
        #     break

# images = np.concatenate(images_list)
# labels = np.concatenate(labels_list)
# preds = np.concatenate(preds_list)

# Ignore index
# ignore_pixel = labels == 255
# preds[ignore_pixel] = 0
# labels[ignore_pixel] = 0

# Plot
"""
fig, axes = plt.subplots(4, 3, figsize=(12, 10))
plt.tight_layout()

axes[0, 0].set_title('input image')
axes[0, 1].set_title('prediction')
axes[0, 2].set_title('ground truth')

for ax, img, lbl, pred in zip(axes, images, labels, preds):
    ax[0].imshow(minmax_normalize(img, norm_range=(0, 1), orig_range=(-1, 1)))
    ax[1].imshow(pred)
    ax[2].imshow(lbl)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

plt.savefig('eval.png')
plt.close()
"""
