import torch
from model.unet import UNet, SUNet
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors
color = colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow', 'blue'])
# color = colors.LinearSegmentedColormap.from_list("", ['white', 'red', 'green', 'yellow'])
from natsort import os_sorted
from pathlib import Path
import cv2
import itertools

def load_model(variant, channel, model_path):
    # initialise model
    model = UNet(n_channels=channel, n_classes=5, bilinear=False) if variant == 'basic' else SUNet(n_channels=channel, n_classes=5, bilinear=False)
    # load the weight of the trained model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # switch the model to inference model which freeze the weight of the layer
    model.eval();
    return model

def visualise_predicted(model, image, name):
    img_ndarray = np.array(image)
    img_ndarray = img_ndarray[np.newaxis, ...] if img_ndarray.ndim == 2 else img_ndarray.transpose((2, 0, 1))
    img_ndarray = img_ndarray / 255
    image_tensor = torch.from_numpy(img_ndarray).unsqueeze(0).float()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)[0] 
        probs = probs.data.numpy().transpose((1, 2, 0))#.detach.cpu().numpy()
        mask = np.argmax(probs, axis=2)
        Image.fromarray((mask).astype(np.uint8)).save(f"predicted/mask/{name}.png")
        
        plt.imshow(np.array(image))
        plt.imshow(mask, cmap=color, alpha=0.6)
        plt.axis('off')
        plt.savefig(f"predicted/{name}.png", bbox_inches='tight', pad_inches = 0, dpi=300)
    return mask

def run_single():
    model = load_model(variant='basic', channel=1, model_path='checkpoint/gray-epoch05.pth')
    
    for i in ['f_01340', 'f_01522']:
        img = Image.open(f'data/test_images/{i}.png').convert('RGB')
        mask = visualise_predicted(model, img, i)

run_single()