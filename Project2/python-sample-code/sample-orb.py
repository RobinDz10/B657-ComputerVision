#!/l/python3.5.2/bin/python3
#
# This is just some messy code to show you how to use the
# ORB feature extractor.
# D. Crandall, Feb 2022

import cv2
import numpy as np
 
img = cv2.imread("lincoln.jpg", cv2.IMREAD_GRAYSCALE)

# you can increase nfeatures to adjust how many features to detect 
orb = cv2.ORB_create(nfeatures=1000)

# detect features 
(keypoints, descriptors) = orb.detectAndCompute(img, None)

# put a little X on each feature
for i in range(0, len(keypoints)):
   print("Keypoint #%d: x=%d, y=%d, descriptor=%s, distance between this descriptor and descriptor #0 is %d" % (i, keypoints[i].pt[0], keypoints[i].pt[1], np.array2string(descriptors[i]), cv2.norm( descriptors[0], descriptors[i], cv2.NORM_HAMMING)))
   for j in range(-5, 5):
      img[int(keypoints[i].pt[1])+j, int(keypoints[i].pt[0])+j] = 0 
      img[int(keypoints[i].pt[1])-j, int(keypoints[i].pt[0])+j] = 255 

cv2.imwrite("lincoln-orb.jpg", img)
