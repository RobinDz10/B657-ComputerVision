from PIL import Image


def rotate_injected_form():
    rotation_degree = 1
    injected_form = Image.open('test-images/injected.jpg')
    rotated_injected_form = injected_form.rotate(rotation_degree)
    rotated_injected_form.save('test-images/injected_form_rotated.jpg')


rotate_injected_form()
