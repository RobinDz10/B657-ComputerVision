"""
This opencv imshow() mouse click code is quite complex and not popular
I refer the code from https://stackoverflow.com/questions/60587273/drawing-a-line-on-an-image-using-mouse-clicks-with-python-opencv-library
the logic of 'capturing keyboard and mouse action' refers from the link above
Except these 'keyboard hook' code, other code is solely my work.
"""

import cv2
import run as runpy


class DrawLineWidget(object):
    def __init__(self, img_path,program_language,program_debug, program_direct):
        self.original_image = cv2.imread(img_path)
        self.clone = self.original_image.copy()
        self.program_language=program_language
        self.program_debug=program_debug
        self.program_direct=program_direct

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]
        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            # cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            column_x = self.image_coordinates[0][0]
            row_y = self.image_coordinates[0][1]

            dig_right = min(100, self.clone.shape[1] - column_x)
            dig_left = min(100, column_x)
            dig_up = min(30, row_y)
            dig_down = min(30, self.clone.shape[0] - row_y)

            print("right:%d, left:%d, up:%d, down:%d" % (dig_right, dig_left, dig_up, dig_down))
            temp_img=self.clone.copy()
            cv2.line(temp_img, (column_x-dig_left,row_y-dig_up), (column_x+dig_right,row_y-dig_up), (0, 255, 0), 3)
            cv2.line(temp_img, (column_x-dig_left,row_y-dig_up), (column_x-dig_left,row_y+dig_down), (0, 255, 0), 3)
            cv2.line(temp_img, (column_x-dig_left,row_y+dig_down), (column_x+dig_right,row_y+dig_down), (0, 255, 0), 3)
            cv2.line(temp_img, (column_x+dig_right,row_y-dig_up), (column_x+dig_right,row_y+dig_down), (0, 255, 0), 3)
            # cv2.line(temp_img, (64,482), (200,200), (0, 255, 0), 3)
            # temp_img.fill(0)
            cv2.imshow("image", temp_img)
            cv2.waitKey(1)


            crop_img = self.clone[row_y - dig_up:row_y + dig_down, column_x - dig_left:column_x + dig_right]
            cv2.imwrite('crop.png', crop_img)
            runpy.run('crop.png', offset=(row_y - dig_up, column_x - dig_left), mouse_mode=True, program_language=self.program_language,program_debug=self.program_debug, program_direct=self.program_direct)

            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone


def run_imgdraw(img_path,program_language,program_debug, program_direct):
    draw_line_widget = DrawLineWidget(img_path,program_language,program_debug, program_direct)
    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
