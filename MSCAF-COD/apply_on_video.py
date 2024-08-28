import torch
import torch.nn.functional as F
import cv2
from lib.MSCSFNet import Network
import torchvision.transforms as transformers
from PIL import Image
from skimage import img_as_ubyte
import os


# load model
PATH = "_Net_epoch_best.pth"
model = Network().cuda()
model.load_state_dict(torch.load(PATH))
model.eval()

save_path = "C:/Maral/MSCAF-COD/tested_videos/"

# load a single image


def frame_load(path):
    img = Image.open(path)
    img = img.convert("RGB")
    transform = transformers.Compose([
        transformers.Resize((352, 352)),
        transformers.ToTensor(),
        transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test = transform(img).unsqueeze(0)
    test = test.cuda()
    return test

# load a video


def preprocess_from_video(frame):
    img = Image.fromarray(frame)
    img = img.convert("RGB")
    transform = transformers.Compose([
        transformers.Resize((352, 352)),
        transformers.ToTensor(),
        transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test = transform(img).unsqueeze(0)
    test = test.cuda()
    return test


# Applying the model on the video
# wichtig size = [height, width] not [width, height]
def apply_model(frame, shape):
    cam, _1, _2, _3, _4 = model(frame)
    cam = F.upsample(cam, size=shape,
                     mode='bilinear', align_corners=True)
    cam = cam.sigmoid().data.cpu().numpy().squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return img_as_ubyte(cam)


def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# #### DEBUG
# image_path = "C:/Maral/Military-Camouflage-MHCD2022/JPEGImages/000565.jpg"
# temp = cv2.imread(image_path)
# result = preprocess_from_video(temp)
# result = apply_model(result)

# save_path = "test_debug.png"
# imageio.imsave(save_path, result)
# Image.open(save_path)


if __name__ == '__main__':
    input_video_name = "video/infantry_1.avi"
    output_video_name = input_video_name + "_mask.avi"

    cap = cv2.VideoCapture(input_video_name)

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    frame_size = (width, height)

    out = cv2.VideoWriter(output_video_name, fourcc, fps, frame_size, False)

    frame_number = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            temp = preprocess_from_video(frame)
            result = apply_model(temp, [height, width])
            # print(result)
            # print(result.shape)
            save_path = save_path + "%d.jpg" % frame_number
            cv2.imwrite(save_path, result)
            # # # # path = "Results/nc4k/pic_%d.png" %frame_number
            # help = cv2.imread(save_path)
            # print(help.shape)
            if result is None:
                print("Error reading image {save_path}.")
                continue
            else:
                out.write(result)
            frame_number += 1
        else:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
