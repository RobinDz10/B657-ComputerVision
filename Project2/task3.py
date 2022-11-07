# Some ideas and code are referred from https://github.com/linrl3/Image-Stitching-OpenCV
import cv2 as cv
import numpy as np
import sys
import os

def ransac_algorithm(image1, image2):
    """
    This function takes two images as its parameters and return the transformation relationship between two input images based on RANSAC

    This function served for several propose:
    1. extract keypoints from two input images
    2. Calculate the transformation relationship between two input images
    It will return the transformation relationship between two input images
    """
    orb = cv.ORB_create(nfeatures=1000)
    (keypoints1, descriptor1) = orb.detectAndCompute(image1, None)
    (keypoints2, descriptor2) = orb.detectAndCompute(image2, None)
    matcher = cv.BFMatcher()
    pre_matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    # For keypoints matching
    post_points, post_points_matches = [], []
    # Draw transformation relationship image between image 1 and image2, based on seperate keypoints
    for matches1, matches2 in pre_matches:
        # Set corrective ratio = 0.85, this value is changeable, for other test
        if 0.85 * matches2.distance > matches1.distance:
            post_points.append((matches1.trainIdx, matches1.queryIdx))
            post_points_matches.append([matches1])
    # Draw transformation relationship image between image 1 and image2, based on seperate keypoints
    image3 = cv.drawMatchesKnn(image1, keypoints1, image2, keypoints2, post_points_matches, None, flags=2)
    cv.imwrite('points_matching.jpg', image3)
    test_tri_similarity = [[1, 1, 1], [0.5, 0.5], [2]]
    similarity_matrix = np.array([])
    # From line 34 to line 46, I referred the source code from https://github.com/linrl3/Image-Stitching-OpenCV
    for row_index in range(similarity_matrix.shape[0]):
        dest = np.array(test_tri_similarity[row_index])
        np.vstack((dest, similarity_matrix))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    # Set minimum matching points pairs = 10, this value is changeable, for other test
    if len(post_points) > 10:
        # Calculate good keypoints in image 1 and image 2
        image1_keypoints = np.float32([keypoints1[i].pt for (_, i) in post_points])
        image2_keypoints = np.float32([keypoints2[i].pt for (i, _) in post_points])
        # Use RANSAC algorithm to calculate transformation relationship between keypoints in image 1 and keypoints in image 2
        (H, status) = cv.findHomography(image2_keypoints, image1_keypoints, cv.RANSAC, 5.0)
    return H


def create_mask(image1, image2, version):
    """
    This function is used for creating mask background for the input two images
    """
    # default smoothing window size = 0, no "black block" between two stitched images
    image1_height, image1_width, image2_width = image1.shape[0], image1.shape[1], image2.shape[1]
    panoramic_height, panoramic_width = image1_height, image1_width + image2_width
    # Establish mask background based on two images relative positions
    # From line 58 to line 68, I referred the source code from: https://github.com/linrl3/Image-Stitching-OpenCV
    mask = np.zeros((panoramic_height, panoramic_width))
    if version == 'ImageBaseLeft':
        # The smoothing window size is changeable, based on testing cases and requirements
        mask[:, image1_width:image1_width] = np.tile(np.linspace(1, 0, 2 * 0).T, (panoramic_height, 1))
        mask[:, :image1_width] = 1
    else:
        # The smoothing window size is changeable, based on testing cases and requirements
        mask[:, image1_width:image1_width] = np.tile(np.linspace(0, 1, 2 * 0).T, (panoramic_height, 1))
        mask[:, image1_width:] = 1

    return cv.merge([mask, mask, mask])


def panoramic_stitching(image1, image2, reverse=False):
    """
    This function is used for stitching (blending) two images into one single image
    """
    # Get result from ransac management
    H = ransac_algorithm(image1, image2)
    image1_height, image1_width, image2_width = image1.shape[0], image1.shape[1], image2.shape[1]
    output_height, output_width = image1_height, image1_width + image2_width
    # The first mask here used for "base image on the left"
    panoramic1 = np.zeros((output_height, output_width, 3))
    # From line 82 to line 98, I referred the source code from: https://github.com/linrl3/Image-Stitching-OpenCV
    first_mask = create_mask(image1.copy(), image2.copy(), version='ImageBaseLeft')
    # The second mask here used for "base image on the right"
    panoramic1[0:image1_height, 0:image1_width, :] = image1
    panoramic1 *= first_mask
    second_mask = create_mask(image1.copy(), image2.copy(), version='ImageBaseRight')
    panoramic2 = cv.warpPerspective(image2, H, (output_width, output_height)) * second_mask
    # Stitching two images into one image
    result = panoramic1 + panoramic2
    rows, cols = np.where(result[:, :, 0] != 0)
    res_minRow, res_minCol, res_maxRow, res_maxCol = min(rows), min(cols), max(rows) + 1, max(cols) + 1
    final_result = result[res_minRow: res_maxRow, res_minCol: res_maxCol, :]
    test_tri_similarity = [[1, 1, 1], [0.5, 0.5], [2]]
    similarity_matrix = np.array([])
    for row_index in range(similarity_matrix.shape[0]):
        dest = np.array(test_tri_similarity[row_index])
        np.vstack((dest, similarity_matrix))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    # if final_result.shape[0] * final_result.shape[1] < image1.shape[0] * image1.shape[1] + image2.shape[0] * image2.shape[1]:
    #     return panoramic_stitching(image2, image1)
    # else:
    #     return final_result

    return final_result


def run(argv):
    """
    This function is the interface used for accepting parameters from terminal
    """
    img1 = cv.imread(argv[2])
    img2 = cv.imread(argv[3])
    img = panoramic_stitching(img1, img2)
    if img.shape[1] <= min(img1.shape[1],img2.shape[1]):
        command = './a2 task3 '+argv[2]+' '+argv[3]+' '+argv[4]
        print(command)
        os.system(command)
    else:
        cv.imwrite(argv[4], img)
