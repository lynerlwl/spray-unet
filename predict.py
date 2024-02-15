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

def generate_contours(mask, class_value):
    selected_pixel = mask.copy()
    selected_pixel[selected_pixel != class_value] = 0
    selected_pixel[selected_pixel == class_value] = 255
    # selected_pixel = selected_pixel.astype(np.uint8)
    # Image.fromarray(selected_pixel).save('test1.png')
    # Applying binary thresholding on the image  
    # _, binary = cv2.threshold(selected_pixel, 50, 255, cv2.THRESH_BINARY_INV) 
    # Grab the contours in the image
    contours, hierarchy = cv2.findContours(selected_pixel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_centroid(cnt):
    M = cv2.moments(cnt)
    cX = int(M['m10']/M['m00'])
    cY = int(M['m01']/M['m00'])
    # Plot circle to check if it is overlap
    # if plot == True:
    #     cv2.circle(image, (cX, cY), 30, (0, 0, 0), 1) # draw circle for initial checking, can be skiped
    return (cX, cY)

def calculate_distance(a, b):
    euclidean_distance = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    return euclidean_distance 

def remove_overlap(contours, r):
    selected_contours = [c for c in contours if 50 <= cv2.arcLength(c,True)]
    point_list = [calculate_centroid(i) for i in selected_contours]
    point_combined = itertools.combinations(point_list, 2)
    d_list = ['y' for a,b in point_combined if calculate_distance(a,b) <= (r*2)]
    count = len(contours) - len(d_list)
    return count

def run_single():
    model = load_model(variant='basic', channel=1, model_path='checkpoint/gray-epoch05.pth')
    
    for i in ['f_01340', 'f_01522']:
        img = Image.open(f'data/test_images/{i}.png').convert('L')
        # img = Image.open(f'data/test_images/{i}.png').convert('RGB')
        mask = visualise_predicted(model, img, i)
    
mask = Image.open(f'predicted/mask/f_01522.png')
np.unique(np.array(mask))
    
    
def run_sequences():
    model = load_model(variant='basic', channel=3, model_path='checkpoint/basic-loss=0.64_dice=0.64.pth')#loss=0.64_dice=0.64.pth shift-equivariant/rgb-loss=0.81_dice=0.66.pth
    images_dir = Path('data/generated_sequences/')
    ids = [filename.stem for filename in os_sorted(images_dir.glob('*.png' or '*.jpg'))]
    for i in ids:
        img = Image.open(f'{images_dir}/{i}.png').convert('RGB')
        mask = visualise_predicted(model, img, i)
        
        selection = ['f_01340', 'f_01522']
        mask = Image.open(f"predicted/rgb-loss=0.81_dice=0.66/mask/{selection[1]}.png")
        mask = np.array(mask)
        if 1 in np.unique(mask):
            contours = generate_contours(mask, class_value = 1)
            drop_counts = len(contours)
        if 2 in np.unique(mask):
            contours = generate_contours(mask, class_value = 2)
            lig_counts = remove_overlap(contours, 10)
        else:
            drop_counts, lig_counts = (0, 0)
        
    # with open('object_count.csv', 'w') as f:
    #     f.write("img_name,drop_counts,lig_counts")
        
        with open('object_count.csv', 'a') as f:
            f.write(f"\n{i},{drop_counts},{lig_counts}")
