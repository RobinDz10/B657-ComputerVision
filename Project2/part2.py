from PIL import Image
import numpy as np
import math

def proj_transform(img_in, mat):
    # convert image to numpy ndarray
    data_in = np.asarray(img_in)
    cols = data_in.shape[1]
    rows = data_in.shape[0]
    bands = data_in.shape[2]

    # homogeneous coordinates of four corners of input image in original coordinate system
    src_corners = np.asarray([[0, 0, 1],
                          [cols-1, 0, 1],
                          [cols-1, 1-rows, 1],
                          [0, 1-rows, 1]]).transpose()
    # homogeneous coordinates of four corners after warping
    dst_corners = np.dot(mat, src_corners)
    # retrieve the 2d coordinates in original coordinate system
    dst_corners_xy = dst_corners[:2,:].copy()
    dst_corners_xy[0] = np.divide(dst_corners_xy[0], dst_corners[2])
    dst_corners_xy[1] = np.divide(dst_corners_xy[1], dst_corners[2])
    # calculate the resolution/size of output image after warping
    dst_range = np.max(dst_corners_xy, axis=1) - np.min(dst_corners_xy, axis=1)
    dst_range = dst_range.astype(int) + 1
    # store the difference in origin of two coordinate systems: one for original image, one for warped image
    offset_xy = np.asarray([np.min(dst_corners_xy[0]), np.max(dst_corners_xy[1])]).reshape(-1)

    # inverse transformation
    mat_inv = np.linalg.inv(mat)
    # inverse warping
    data_out = np.zeros((dst_range[1], dst_range[0], bands), dtype=np.uint8)
    for r in np.arange(dst_range[1]):
        for c in np.arange(dst_range[0]):
            # homogeneous coordinate of pixel in the coordinate system of resulting image
            dst_xyw = np.asarray([c, r*-1, 1], dtype=float).transpose()
            # change the coordinate system to the coordinate system of original image
            dst_xyw[:2] += offset_xy
            # apply the inverse warping
            src_xyw = np.dot(mat_inv, dst_xyw)
            # retrieve the 2d coordinate
            src_xy = src_xyw[:2] / src_xyw[2]
            # convert coordinate to row and column numbers
            c0 = src_xy[0]
            r0 = src_xy[1]*-1
            # resampling if corresponding pixel is within the extent of original image
            if (c0 >= 0 and c0 <= cols-1) and (r0 >= 0 and r0 <= rows-1):
                data_out[r,c] = data_in[int(r0), int(c0)]
            # fill the pixel with black otherwise
            else:
                data_out[r,c] = 0

    print(data_out.shape)
    img_out = Image.fromarray(data_out)
    return img_out

if __name__ == "__main__":
    img_path = "part2-images/lincoln.jpg"
    out_path = "lincoln_trans.jpg"
    # transformation matrix
    mat = np.asarray([[-0.153, 1.44, 58],
                      [0.907, 0.258, -182],
                      [-0.000306, 0.000731, 1]])
    mat_trans = np.asarray([[1, 0, 100],
                            [0, 1, 100],
                            [0, 0, 1]])
    mat_scale = np.asarray([[0.5, 0, 0],
                            [0, 0.5, 0],
                            [0, 0, 1]])
    mat_rotate = np.asarray([[math.cos(math.radians(45)), -1*math.sin(math.radians(45)), 0],
                             [math.sin(math.radians(45)), math.cos(math.radians(45)), 0],
                             [0, 0, 1]])
    mat_shear = np.asarray([[1, 1, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    img_in = Image.open(img_path)
    img_out = proj_transform(img_in, mat)
    img_out.save(out_path)
    img_in.close()
    img_out.close()