import sys

import cv2
import numpy as np
import pytesseract as ts
import string_handle
import opencv_imgdraw
import sys

import os


def is_text_black(roi):
    """
    detect input img (cropped small image)'s properity: whether it's white-based-color-text or black-based-color-text
    :param roi: input image, should be greyscale
    :return: isBlack
    """
    global is_text_black_count
    is_text_black_count += 1
    ret, threshold_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(str(is_text_black_count) + '_threshold.png', threshold_roi)
    black_pixel = np.count_nonzero(threshold_roi == 0)
    white_pixel = np.count_nonzero(threshold_roi == 255)

    # if text is black, then black pixel count should be much smaller than white pixel count
    return black_pixel < white_pixel


def run(img_path, offset=(0, 0), mouse_mode=False, real_original_img_path='',program_debug=False, program_language='eng', program_direct=False):
    # offset: (row_y, column_x)
    global is_text_black_count
    is_text_black_count = 0
    img = cv2.imread(img_path)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    # cv2.imwrite('output.png', img)

    print(offset)

    if mouse_mode:
        detect(img, img_orig, cv2.imread('remove_privacy.png'), offset,program_language=program_language, program_debug=program_debug, program_direct=program_direct)
    else:
        detect(img, img_orig, cv2.imread(real_original_img_path), offset, program_language=program_language, program_debug=program_debug, program_direct=program_direct)


def preprocess(gray,program_debug):
    # to clarity: the function(mainly the kernel size) is referred from https://github.com/lzmisscc/Form-Detection,
    # in order to quickly check if our project is runnable.
    # I write the code using the same algorithm in my own style
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    dilation = cv2.dilate(binary, element2, iterations=1)

    erosion = cv2.erode(dilation, element1, iterations=1)

    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    if program_debug:
        cv2.imwrite("binary.png", binary)
        cv2.imwrite("dilation.png", dilation)
        cv2.imwrite("erosion.png", erosion)
        cv2.imwrite("dilation2.png", dilation2)

    return dilation2, dilation


def findTextRegion(img):
    # to clarity: the function(mainly the logic to judge text area) is referred from https://github.com/lzmisscc/Form-Detection,
    # in order to quickly check if our project is runnable.
    # I write the code using the same algorithm in my own style
    region = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area < 1000):
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if (height > width * 1.2):
            continue
        region.append(box)
    return region


def findAvatarRegion(img):
    region = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area < 1000):
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = width / height
        if ratio > 1.5 or ratio < 0.7:
            continue
        region.append(box)
    return region


def detect(grey_img, original_img, real_original_img, offset,program_language, program_debug,program_direct):
    print("debug: %s" % str(program_debug))
    print("lang: %s" % str(program_language))
    print("direct: %s"%program_direct)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    debug_color_img = original_img.copy()
    privacy_remove_img = original_img.copy()

    # to clarity: the next 20~25 lines' code is referred from https://github.com/lzmisscc/Form-Detection,
    # in order to quickly check if our project is runnable.
    # I write the code using the same algorithm in my own style

    dilation, dilation_avatar = preprocess(grey_img, program_debug)

    region = findTextRegion(dilation)
    region_avatar = findAvatarRegion(dilation_avatar)
    cv2.drawContours(debug_color_img, region_avatar, -1, (255, 0, 0), 2)

    for box in region_avatar:
        xmax, ymax = box.max(axis=0)
        xmin, ymin = box.min(axis=0)
        # roi = grey_img[ymin:ymax, xmin:xmax]
        x_mid = (xmax + xmin) // 2
        y_mid = (ymax + ymin) // 2
        if x_mid / (debug_color_img.shape[1]) < 0.15 or x_mid / (debug_color_img.shape[1]) > 0.85:
            cv2.rectangle(privacy_remove_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)

    contour_img = 0
    for box in region:
        cv2.drawContours(debug_color_img, [box], 0, (0, 255, 0), 2)
        contour_img += 1
        xmax, ymax = box.max(axis=0)
        xmin, ymin = box.min(axis=0)
        roi = grey_img[ymin:ymax, xmin:xmax]

        if not is_text_black(roi):
            roi = 255 - roi

        ret, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if program_debug:
            cv2.imwrite(str(contour_img) + '.jpg', roi)

        # begin OCR
        # text_full = ts.image_to_string(roi, 'chi_sim')
        # print(text)

        # if '114' in text or '2566' in text:
        # print(text.strip())

        data = ts.image_to_data(roi, output_type='dict', lang=program_language)
        boxes = len(data['level'])
        # --------
        text_full = ""
        text_full_list = []
        for i in range(boxes):
            text_full += data['text'][i]  # not used but just left here for reference
            text_full_list.append(data['text'][i])
        # ----------
        for i in range(boxes):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # Draw box
            # if '118' in data['text'][i] or '2566' in data['text'][i] or '114' in data['text'][i]:
            #     cv2.rectangle(privacy_remove_img, (x+xmin, y+ymin), (x+xmin + w, y+ymin + h), (255, 0, 0), -1)
            # else:
            #     cv2.rectangle(privacy_remove_img, (x+xmin, y+ymin), (x+xmin + w, y+ymin + h), (0,255,0), 1)
            # data['text'][i] is the string to check and detect
            # print(data['text'][i].strip(), end='')

            offset2 = 0
            if i == 0:
                offset2 = 0
            else:
                for j in range(0, i):
                    offset2 = offset2 + len(data['text'][j]) + 1

            # if string_handle.detect_number(text_full,data['text'][i],offset2):
            if string_handle.detect_privacy(text_full_list, i):
                cv2.rectangle(privacy_remove_img, (x + xmin, y + ymin), (x + xmin + w, y + ymin + h), (255, 0, 0),
                              -1)
            # else:
            cv2.rectangle(debug_color_img, (x + xmin, y + ymin), (x + xmin + w, y + ymin + h), (0, 155, 155),
                          1)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)

    # use privacy_remove_img to 'COVER' real original image
    # offset: (row_y, column_x)
    for row_index in range(0, privacy_remove_img.shape[0]):
        for col_index in range(0, privacy_remove_img.shape[1]):
            real_original_img[row_index + offset[0], col_index + offset[1]] = privacy_remove_img[row_index, col_index]


    cv2.imwrite("remove_privacy.png", real_original_img)


    if program_debug:
        cv2.imwrite("debug_contour.png", debug_color_img)

    if not program_direct:
        print("There should be an image-viewer window, please check your windows")
        opencv_imgdraw.run_imgdraw('remove_privacy.png',program_language=program_language,program_debug=program_debug, program_direct=program_direct)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # exit(1)


def print_usage():
    print("""
            python3 run.py [options] -i file
            options:
                --debug: save all temporary files (debug only, could let program run much slower)
                --help: print help function
                --lang [language]: specify pytesseract languages in your system(list below), default: 'eng'
                --direct: don't show interactive window after completed
            """)
    print(ts.get_languages())


if __name__ == '__main__':
    # global options
    program_language = 'eng'
    program_debug = False
    program_direct=False
    # end global options
    is_text_black_count = 0
    # run('test5.jpg')
    args = sys.argv
    # help function
    if (len(args) == 2 and args[1] == '--help') or len(args) == 1:
        print_usage()

    for i in range(0, len(args)):
        if args[i] == '--debug':
            program_debug = True
        if args[i] == '--lang':
            if i < len(args):
                program_language = args[i + 1]
            else:
                print_usage()
        if args[i]=='--direct':
            program_direct=True



    index_file = -1
    for i in range(0, len(args) - 1):
        if args[i] == '-i':
            index_file = i + 1
            break

    if index_file != -1 and os.path.exists(args[index_file]):
        run(img_path=args[index_file], real_original_img_path=args[index_file],program_debug=program_debug, program_language=program_language, program_direct=program_direct)
    else:
        print('invalid input, please see the program\'s help via $ python3 run.py --help')
        print_usage()
