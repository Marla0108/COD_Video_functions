import cv2
import os
from natsort import natsorted

image_folder = "C:/Maral/test_video_pictures"
video_path = "C:/Maral/Models/MSCAF-COD-master/video/"

video_name = "infantry_1.avi"
video = video_path + video_name


images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = natsorted(images)

# #Resize images to a common size
# def resize_image(image, size):
#     return cv2.resize(image, size, interpolation= cv2.INTER_AREA)

# #Determine the common size for the video
# common_size = (352, 352)

# Determine the width and height from the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
target = width, height

# Create a video with VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# Framerate
video = cv2.VideoWriter(video, fourcc, 15, target)


for img in images:
    img_path = os.path.join(image_folder, img)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error reading image {img_path}.")
        continue
    else:
        # resized_frame = resize_image(frame, common_size)
        video.write(frame)
        print(frame.shape)

video.release()


print("The video is successfully saved")
cv2.destroyAllWindows()
