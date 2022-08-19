import os
import numpy as np
import cv2
import glob
# files_dir="/home/u20/d2/code/RGBTCrowdCounting/imgs/*"
files_dir = '/home/u20/d2/code/RGBTCrowdCounting/DroneRGBT/save/train/*RGB.jpg'
files =glob.glob(files_dir) #+"*T.jpg") # os.listdir(files_dir)

R = 0.
G = 0.
B = 0.
R_2 = 0.
G_2 = 0.
B_2 = 0.
N = 0

for file in files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    h, w, c = img.shape
    N += h*w

    R_t = img[:, :, 0]
    R += np.sum(R_t)
    R_2 += np.sum(np.power(R_t, 2.0))

    G_t = img[:, :, 1]
    G += np.sum(G_t)
    G_2 += np.sum(np.power(G_t, 2.0))

    B_t = img[:, :, 2]
    B += np.sum(B_t)
    B_2 += np.sum(np.power(B_t, 2.0))
RGB=np.array([R,G,B])/N
RGB_2=np.array([R_2,G_2,B_2])/N

R_std = np.sqrt(RGB_2 - np.multiply(RGB,RGB))

print(RGB,RGB/255.0)
# print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))
print(R_std,R_std/255.0)
# print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean/255.0, G_mean/255.0, B_mean/255.0))
# print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))
