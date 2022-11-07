
import os

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
from os import walk
import sys

"""
This function handles image clustering prediction results and clustering accuracy.
"""
def error_test(imgList, imgCluster):
    dict = {}
    for i in range(len(imgCluster)):
        key = imgCluster[i]
        dict[key] = []

    for i in range(len(imgCluster)):
        dict[imgCluster[i]].append(imgList[i])

    list3 = []
    error = 0
    for key in dict:
        list4 = dict[key]
        dict1 = {}
        for item in list4:
            val = item.split('_')[0]
            if val not in dict1.keys():
                dict1[val] = 1
            else:
                dict1[val] += 1
        maxcount = 0
        maxkey = ''
        for key in dict1:
            if dict1[key] > maxcount:
                maxcount = dict1[key]
                maxkey = key
        error += (len(list4) - maxcount)

    print("test numbers in total: ", len(imgList))
    print("error numbers in total: ", error)
    print("correct numbers in total: ", len(imgList) - error)
    print("error rate: ", (error / len(imgList)))
    print("correct rate: ", (1 - error / len(imgList)))

"""
the main entrance of matching two images and return a series of data. E.g. similatiry matrix, clustering result.
"""
def match_two_images(img_path1, img_path2):
    """
    Code for orb
    """
    def orb(img):
        # you can increase nfeatures to adjust how many features to detect
        orb = cv2.ORB_create(nfeatures=50)

        # detect features
        (keypoints, descriptors) = orb.detectAndCompute(img, None)

        # val=[[row,column,descriptor]]
        # val=[[integer,integer,np.ndarray()]]
        val = []

        # put a little X on each feature
        for i in range(0, len(keypoints)):
            # print("Keypoint #%d: x=%d, y=%d, distance between this descriptor and descriptor #0 is %d" % (
            #     i,
            #     keypoints[i].pt[0],
            #     keypoints[i].pt[1],
            #     cv2.norm(descriptors[0], descriptors[i], cv2.NORM_HAMMING)))
            column = int(keypoints[i].pt[0])
            row = int(keypoints[i].pt[1])
            val.append([row, column, descriptors[i]])

        return val

    """ 
    for each feature point in image: find the nearest feature point and second nearest feature point in the other image
    """
    def normalize_nearest_dist(f1, val2):
        # distances=[[row,column,distance between f1 and (row,colum)]]
        distances = []
        for item in val2:
            distances.append([item[0], item[1], cv2.norm(f1, item[2], cv2.NORM_HAMMING)])
        distances = np.array(distances)
        distances = distances[distances[:, 2].argsort()]
        norm_dist = distances[0][2] / distances[1][2]
        return_val = (distances[0][0], distances[0][1], norm_dist)
        return return_val

    print(img_path1 + img_path2)
    img = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    # gaussian blur code copy ref: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
    # kernel = np.ones((5, 5), np.float32) / 25
    # img = cv2.filter2D(img, -1, kernel)
    # cv2.imwrite("temp_blur.jpg", img)

    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    # gaussian blur code copy ref: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
    # kernel = np.ones((5, 5), np.float32) / 25
    # img2 = cv2.filter2D(img2, -1, kernel)
    # cv2.imwrite("temp_blur2.jpg", img2)

    # val=[[row,column,descriptor]]
    # val=[[integer,integer,np.ndarray()]]
    val1 = orb(img)
    val2 = orb(img2)

    # combine two matrices
    if img.shape[0] > img2.shape[0]:
        # img is taller
        zero_padding = np.zeros((img.shape[0] - img2.shape[0], img2.shape[1]))
        padding_img = np.concatenate((img2, zero_padding))
        new_img = np.column_stack((img, padding_img))
    else:
        # img2 is taller
        zero_padding = np.zeros((img2.shape[0] - img.shape[0], img.shape[1]))
        padding_img = np.concatenate((img, zero_padding))
        new_img = np.column_stack((padding_img, img2))

    cv2.imwrite("concat_img.jpg", new_img)

    dist_lst = []
    matched_count = 0
    for val in val1:
        nearest_val = normalize_nearest_dist(val[2], val2)
        img1_row = int(val[0])
        img1_column = int(val[1])
        img2_row = int(nearest_val[0])
        img2_column = int(nearest_val[1])
        norm_dist = nearest_val[2]
        # calculate new row and column
        # left is always img, right is always img2

        # opencv:X should be column, opencv:Y should be row
        if norm_dist < 0.9:
            new_img = cv2.line(new_img, (img1_column, img1_row), (img2_column + img.shape[1], img2_row), (0, 255, 0),
                               1)
            matched_count += 1
        dist_lst.append(norm_dist)

    cv2.imwrite((img_path1 + img_path2 + '.jpg').replace("/", ""), new_img)
    #
    #
    # plt.figure()
    # plt.hist(dist_lst)
    # plt.savefig('plt.png')

    return 1 - np.average(np.array(dist_lst))

""" 
main entrance of running the program: iterating the file list and match every two images' pairs. 
Get the result as a similarity matrix and use sklearn-SpectralClustering to cluster the result.
"""
def run_astroid(filenames, k_cluster, output_file):
    filenames.sort()
    tri_similarity = []
    clip = len(filenames)
    # clip = 5
    for i in range(0, clip):
        similarity_vector = []
        for j in range(i + 1, clip):
            similarity = match_two_images(filenames[i], filenames[j])
            similarity_vector.append(similarity)
        tri_similarity.append(similarity_vector)

    # %%
    print(tri_similarity)

    similarity_matrix = np.ones((clip, clip))
    for row_index in range(similarity_matrix.shape[0]):
        dest = np.array(tri_similarity[row_index])
        similarity_matrix[row_index][row_index + 1:] = dest

    from sklearn.cluster import SpectralClustering

    similarity_matrix[np.isnan(similarity_matrix)] = 0
    # print(similarity_matrix)
    # print(filenames)
    res = SpectralClustering(k_cluster).fit_predict(similarity_matrix)

    output = []
    for i in range(0, len(res)):
        output.append((filenames[i] + ": #" + str(res[i])))
    import random
    random.shuffle(output)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as f:
        for item in output:
            f.write("%s\n" % item)
    f.close()

    error_test(filenames, res)

"""
main entrance of the program
"""
def run(argv):
    # part1 is argv[1]
    k_cluster = int(sys.argv[2])
    output_file = sys.argv[-1]
    file_lst = sys.argv[3:-1]
    # print(k_cluster)
    # print(file_lst)
    # print(output_file)

    run_astroid(file_lst, k_cluster, output_file)
