import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.MSCSFNet import Network
import imageio
from skimage import img_as_ubyte
from torch import nn
import torchvision.transforms as transformers
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352,
                    help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='_Net_epoch_best.pth')
parser.add_argument('--test_save', type=str,
                    default='C:/Maral/Testvideos/masks')
opt = parser.parse_args()

model_path = '_Net_epoch_best.pth'
model = Network().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()


def transform_for_model(frame):
    transform = transformers.Compose([
        transformers.Resize((352, 352)),
        transformers.ToTensor(),
        transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test = transform(img).unsqueeze(0)
    test = test.cuda()
    return test


def apply_model(frame, shape):
    cam, _1, _2, _3, _4 = model(frame)
    cam = F.upsample(cam, size=shape, mode='bilinear', align_corners=True)
    cam = cam.sigmoid().data.cpu().numpy().squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    result = img_as_ubyte(cam)
    return result


test_videos = "C:/Maral/Testvideos/camuflaged_pumas_for_Reinhard/"
save_path = "C:/Maral/Testvideos/masks/"


img_count = 1
for subfolder in os.listdir(test_videos):
    item = os.path.join(test_videos, subfolder)
    # check if it is a file (not a dir)
    if os.path.isfile(item) and item.endswith(".png", ".jpg"):
        img = Image.open(item)
        img = img.convert("RGB")
        width, height = img.size
        image = transform_for_model(item)
        result = apply_model(image, [height, width])
        destination = save_path
        if not os.path.exists(destination):
            os.mkdir(destination)
        path = destination + subfolder
        imageio.imsave(path, result)
        Image.open(path)
    img_count += 1
