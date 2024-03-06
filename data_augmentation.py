from PIL import Image
import cv2

target = "f_01213-edit"

img_ori = Image.open(f"data/images/{target}.png").convert('RGB')
msk_ori = Image.open(f"data/masks/{target}.png")

degree_list = [d * 30 for d in range(0, 12)]

for degree in degree_list:
    img_ori_aug = img_ori.rotate(degree, fillcolor='white')
    file_name = f"data/images/img_rgb_{degree}.png"
    img_ori_aug.save(file_name)
    
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(image, (5,5), 0.5) # parameter: image, kernel_size, sigma
    cv2.imwrite(f"data/images/img_rgb_b_{degree}.png", image)
    
    img_ori_aug = img_ori_aug.convert('L')
    file_name = f"data/images/img_g_{degree}.png"
    img_ori_aug.save(file_name)
    
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5,5), 0.5) # parameter: image, kernel_size, sigma
    cv2.imwrite(f"data/images/img_g_b_{degree}.png", image)
    
    mask_ori_aug = msk_ori.rotate(degree)
    file_name = f"data/masks/img_rgb_{degree}.png"  
    mask_ori_aug.save(file_name)
    file_name = f"data/masks/img_rgb_b_{degree}.png"  
    mask_ori_aug.save(file_name)
    file_name = f"data/masks/img_g_{degree}.png" 
    mask_ori_aug.save(file_name)
    file_name = f"data/masks/img_g_b_{degree}.png" 
    mask_ori_aug.save(file_name)