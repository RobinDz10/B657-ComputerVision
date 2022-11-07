from PIL import Image
import numpy as np

def proj_transform(img_in, mat):
    # convert image to numpy ndarray
    data_in = np.asarray(img_in)
    cols = data_in.shape[1]
    rows = data_in.shape[0]
    bands = data_in.shape[2]

    # inverse transformation
    mat_inv = np.linalg.inv(mat)
    # inverse warping
    data_out = np.zeros((rows, cols, bands), dtype=np.uint8)
    for r in np.arange(rows):
        for c in np.arange(cols):
            # homogeneous coordinate of pixel in the coordinate system of resulting image
            dst_xyw = np.asarray([c, r, 1], dtype=float).transpose()
            # apply the inverse warping
            src_xyw = np.dot(mat_inv, dst_xyw)
            # retrieve the 2d coordinate
            src_xy = src_xyw[:2] / src_xyw[2]
            # convert coordinate to row and column numbers
            c0 = src_xy[0]
            r0 = src_xy[1]
            # resampling if corresponding pixel is within the extent of original image
            if (c0 >= 0 and c0 <= cols - 1) and (r0 >= 0 and r0 <= rows - 1):
                data_out[r, c] = data_in[int(r0), int(c0)]
            # fill the pixel with black otherwise
            else:
                data_out[r, c] = 0

    print(data_out.shape)
    img_out = Image.fromarray(data_out)
    return img_out

def get_trans_matrix(n, img_dst, img_src, points_dst, points_src):
    # convert row and col numbers to xy coordinates
    dst_xy = points_dst.copy()
    #dst_xy[:,1] = img_dst.size[1] - 1 - points_dst[:,1]
    src_xy = points_src.copy()
    #src_xy[:,1] = img_src.size[1] - 1 - points_src[:,1]

    if n == 1:
        mat = np.zeros((3,3), dtype=float)
        mat[0,0] = 1
        mat[1,1] = 1
        mat[2,2] = 1
        mat[0:2,2] = dst_xy - src_xy

    if n == 4:
        # coefficient matrix
        X = np.zeros((2*n, 8), dtype=float)
        p = np.zeros((2*n,))
        # initialize coefficient matrix
        for i in np.arange(n):
            x = src_xy[i,0]
            y = src_xy[i,1]
            x_ = dst_xy[i,0]
            y_ = dst_xy[i,1]
            X[i*2] = np.asarray([x, y, 1, 0, 0, 0, -x*x_, -x_*y])
            X[i*2+1] = np.asarray([0, 0, 0, x, y, 1, -x*y_, -y*y_])
            p[i*2] = x_
            p[i*2+1] = y_
        # solve the equation
        H = np.linalg.solve(X, p)
        # get the transformation matrix
        mat = np.r_[H, 1].reshape(3,3)
    return mat

def run(argv):
    # print(argv)
    #img_path = "part2-images/lincoln.jpg"
    #out_path = "lincoln_trans.jpg"

   # img1_path = "book3.jpg"
   # img2_path = "part2-images/book2.jpg"
   # out_path = "book_output.jpg"
    img1_path=argv[3]
    img2_path=argv[4]


    # open two images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # parameters: number of matched points and their coordinates
    n =int(argv[2])


    arg1=[]
    arg2=[]
    for i in range(6,len(argv)):
        if i%2==0:
            arg1.append([int(argv[i].split(',')[0]), int(argv[i].split(',')[1])])
        else:
            arg2.append([int(argv[i].split(',')[0]), int(argv[i].split(',')[1])])

    # print(arg1)
    # print(arg2)

    points1_cr = np.asarray(arg1)
    points2_cr = np.asarray(arg2)
    #points1_cr = np.asarray([[841, 58]])
    #points2_cr = np.asarray([[1023, 0]])

    # retrieve the transformation matrix
    mat = get_trans_matrix(n, img1, img2, points1_cr, points2_cr)
    print(mat)

    # transform the image using the matrix retrieved above
    img_out = proj_transform(img2, mat)
    img_out.save(argv[5])

    img1.close()
    img2.close()
    img_out.close()

