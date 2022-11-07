#Import the Image and ImageFilter classes from PIL (Pillow)
from PIL import Image
from PIL import ImageFilter
import sys
import random

if __name__ == '__main__':
    # Load an image 
    im = Image.open(sys.argv[1])

    # Check its width, height, and number of color channels
    print("Image is %s pixels wide." % im.width)
    print("Image is %s pixels high." % im.height)
    print("Image mode is %s." % im.mode)

    # Pixels are accessed via an (X,Y) tuple.
    # The coordinate system starts at (0,0) in the upper left-hand corner,
    # and increases moving right (first coordinate) and down (second coordinate).
    # So it's a (col, row) indexing system, not (row, col) like we're used to
    # when dealing with matrices or 2d arrays.
    print("Pixel value at (10,10) is %s" % str(im.getpixel((10,10))))
    
    # Pixels can be modified by specifying the coordinate and RGB value
    # (255, 0, 0) is a pure red pixel.
    im.putpixel((10,10), (255, 0, 0))
    print("New pixel value is %s" % str(im.getpixel((10,10))))

    # Let's create a grayscale version of the image:
    # the "L" means there's only a single channel, "Lightness"
    gray_im = im.convert("L")
    
    # Create a new blank color image the same size as the input
    color_im = Image.new("RGB", (im.width, im.height), color=0)
    gray_im.save("gray.png")
    
    # Highlights any very dark areas with yellow.
    for x in range(im.width):
        for y in range(im.height):
            p = gray_im.getpixel((x,y))
            if p < 5:
                (R,G,B) = (255,255,0)
                color_im.putpixel((x,y), (R,G,B))
            else:
                color_im.putpixel((x,y), (p,p,p))

    # Show the image. We're commenting this out because it won't work on the Linux
    # server (unless you set up an X Window server or remote desktop) and may not
    # work by default on your local machine. But you may want to try uncommenting it,
    # as seeing results in real-time can be very useful for debugging!
    color_im.show()

    # Save the image
    color_im.save("output.png")

    # This uses Pillow's code to create a 5x5 mean filter and apply it to
    # our image. In the lab, you'll need to write your own convolution code (using
    # "for" loops, but you can use Pillow's code to check that your answer is correct.
    # Since the input is a color image, Pillow applies the filter to each
    # of the three color planes (R, G, and B) independently.
    box = [1]*25
    result = color_im.filter(ImageFilter.Kernel((5,5),box,sum(box)))
    result.show()
    result.save("output2.png")
