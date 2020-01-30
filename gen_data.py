import cv2
import os
import numpy as np
import csv
import time

"""
Use to convert custom images to a csv that can be passed to train.py
images should be stored in datasets/images/
the first character of the file name is the images y value
ie. image of sign language "a" should be named "a.jpg" or "a2.jpg", etc.
"""
print("Generating csv from images...")
IMAGEDIR = "./datasets/images/"
with open(f"./datasets/sign_test_{time.time()}.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([0] * 785)
    for img in os.listdir(IMAGEDIR):
        img_array = cv2.imread(os.path.join(IMAGEDIR, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (28, 28))
        img_array = np.array(img_array).reshape(-1)
        img_array = np.insert(img_array, 0, ord(img[0].lower())-97)
        writer.writerow(img_array)
    print(f"Created dataset: {file.name}")
