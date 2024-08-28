import numpy as np
import cv2 
import matplotlib.pyplot as plt 
import os 
import imageio

def binary_to_edge(mask): 
    edge = cv2.Canny(mask, 100, 200)
    return edge


def read_and_process_images(input_folder, output_folder, process_function):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in os.listdir(input_folder):
        if file.endswith(".png"):
            img_path = os.path.join(input_folder+file)
            img = cv2.imread(img_path)  
            
            if img is not None:
                processed_img = process_function(img)
                # processed_img = processed_img.astype(np.uint8)*255
                output_path = os.path.join(output_folder + file)
                imageio.imsave(output_path, processed_img)

def process_image(input_dir, output_dir, process_function):
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            for file in files: 
                if file.endswith(".png"):
                    input_path = os.path.join(root, dir)
                    print("input_path",input_path)
                    relative_path = os.path.relpath(input_path, input_dir)
                    print("relative_path", relative_path)
                    output_path = os.path.join(output_dir, relative_path)
                    print("output_path", output_path)
                
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                img_path = os.path.join(input_path + file)
                img = cv2.imread(img_path)
                
                if img is not None: 
                    pro_img = process_function(img)
                    imageio.imsave(output_path + file, pro_img)
                    print("Saved picture in", output_path)
                else: 
                    print("Cant read picture in", img_path)
        # for file in files: 
        #     if file.endswith(".png"):
        #         input_path = os.path.join(root, dirs)
        #         relative_path = os.path.relpath(input_path, input_dir)
        #         output_path = os.path.join(output_dir, relative_path)
                
        #         if not os.path.exists(output_path):
        #             os.makedirs(output_path)
                
        #         img = cv2.imread(input_path + file)
                
        #         if img is not None: 
        #             pro_img = process_function(img)
        #             imageio.imsave(output_path + file, pro_img)
        #             print("Saved picture in", output_path)
                    

ground_truth_folder = 'C:/Maral/Military-Camouflage-MHCD2022/GT/'
edge_folder = "C:/Maral/Military-Camouflage-MHCD2022/Edge/"

input_folder_moca = "C:/Maral/Test_Moca/"
output_folder_moca = "C:/Maral/Test_Moca_Mask/" 

# process_image(input_folder_moca, output_folder_moca, binary_to_edge)

read_and_process_images(ground_truth_folder, edge_folder, binary_to_edge)


# test_input = "C:/Maral/Military-Camouflage-MHCD2022/GT/000002.png"
# test_img = cv2.imread(test_input)
# plt.imshow(test_img)
# plt.axis('off')
# plt.show()


# test_edge = binary_to_edge(test_img)

# plt.imshow(test_edge, cmap= "gray")
# plt.show()

# test_edge = test_edge.astype(np.uint8)*255
# imageio.imsave(edge_folder, test_edge)