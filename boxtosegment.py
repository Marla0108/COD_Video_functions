from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import imageio
from skimage import img_as_ubyte
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import xml.etree.ElementTree as ET  

device = "cuda"
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)

def show_mask(mask):
    color = np.array([255, 255, 255, 0.99])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.axis("off")
    plt.imshow(mask_image)
     
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    

pic_folder = "C:/Maral/Military-Camouflage-MHCD2022/JPEGImages/"
jpeg_pic = "000298.jpg"
image = cv2.imread(pic_folder + jpeg_pic)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis('off')
plt.show()

mask_predicter = SamPredictor(sam)
mask_predicter.set_image(image)

#Single Object: 
box_prompt = np.array([614, 620, 925, 1354])

mask, _, _ = mask_predicter.predict(
    point_coords=None,
    point_labels=None,
    box=box_prompt[None, :],
    multimask_output=False,
)

show_mask(mask)
show_box(box_prompt, plt.gca())
image_array= img_as_ubyte(mask)
save_path = "C:/Maral/Military-Camouflage-MHCD2022/GT/"
name = "000156.png"
imageio.imsave(save_path + name, image_array[0])
print("Saved successfully in", name)


#Multi-Object: 
image1_boxes = torch.tensor([
    [152, 171, 270, 266],
    [171, 350, 256, 480],
    [286, 24, 345, 99]
], device=sam.device)

transformed_boxes = mask_predicter.transform.apply_boxes_torch(image1_boxes, image.shape[:2])

masks_batch, _, _ = mask_predicter.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
for mask in masks_batch:
    show_mask(mask.cpu().numpy())
for box in image1_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()

for mask in masks_batch:
    show_mask(mask.cpu().numpy())

image_array= img_as_ubyte(masks_batch.cpu().numpy())
# new = np.zeros(image_array[0].shape)
# for array in range(len(image_array)):
#     new = new + image_array[array]
    
new = image_array[0] + image_array[1] + image_array[2]
save_path = "C:/Maral/Military-Camouflage-MHCD2022/GT/"
name = "000167.png"
imageio.imsave(save_path + name, new[0])
print("Saved successful in ", name)