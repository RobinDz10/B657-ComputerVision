import imageio
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from kernel_exp import read_ground_truths, apply_kernel_all_channels, get_x_y_from_flatten_arr, \
    get_top_not_in_proximity, get_value_dict, check_proximity

img_width = 1700
img_height = 2200
total_options = 425

vertical_injection_count = int((img_height / (img_height + img_width)) * total_options)
vertical_injection_count_left = int(vertical_injection_count/2)
vertical_injection_count_right = vertical_injection_count - vertical_injection_count_left

horizontal_injection_count = total_options - vertical_injection_count
horizontal_injection_count_up = int(horizontal_injection_count / 2)
horizontal_injection_count_down = horizontal_injection_count - horizontal_injection_count_up

injection_distance = 0.04 # percent

def get_n_points_on_line(start_pt, end_pt, num_points):
    start_x, start_y = start_pt
    end_x, end_y = end_pt

    point_coords = []

    for i in range(num_points):
        coords = [start_x + int(i * (end_x - start_x) / (num_points - 1)), start_y + int(i * (end_y - start_y) / (num_points - 1))]
        point_coords.append(coords)
    return point_coords

def get_injection_locations(width, height, total_lines = 1):
    corner_coords = [[int(height * injection_distance), int(width * injection_distance)], # top left
                     [int(height * injection_distance), width - int(width * injection_distance)], # top right
                     [height - int(height * injection_distance), int(width * injection_distance)], # bottom left
                     [height - int(height * injection_distance), width - int(width * injection_distance)]] # bottom right

    vert_1 = get_n_points_on_line(corner_coords[0], corner_coords[2], vertical_injection_count_left + 2)
    vert_2 = get_n_points_on_line(corner_coords[1], corner_coords[3], vertical_injection_count_right + 2)

    hor_1 = get_n_points_on_line(corner_coords[0], corner_coords[1], horizontal_injection_count_up + 2)
    hor_2 = get_n_points_on_line(corner_coords[2], corner_coords[3], horizontal_injection_count_down + 2)

    hor_1 = hor_1[1:-1]
    hor_2 = hor_2[1:-1]

    return vert_2[1:-1] + hor_2 + vert_1[1:-1] + hor_1 + vert_1[:1] + vert_1[-1:] + vert_2[:1] + vert_2[-1:]

def get_marker_patch(width, height, detect = False): # also returns kernel
    circle_pt_val = -1 if detect else 0
    no_circle_pt_val = 1 if detect else 255

    center = int(width / 2), int(height/2)
    radius = min(width, height) / 4
    patch = np.full((height, width), no_circle_pt_val)
    for i in range(width):
        for j in range(height):
            dist = np.linalg.norm(np.array([abs(center[0] - i), abs(center[1] - j)]))
            if dist < radius:
                patch[i,j] = circle_pt_val
    return patch

def filter_location_by_idxes(locations, idxes):
    filtered_locations = []
    for idx in idxes:
        filtered_locations.append(locations[idx])
    filtered_locations.extend(locations[-4:])
    return filtered_locations

def inject_patches_in_image(image_arr, patch, locations):
    hor_left = int(patch.shape[0] / 2)
    hor_right = patch.shape[0] - hor_left
    ver_left = int(patch.shape[1] / 2)
    ver_right = patch.shape[1] - ver_left

    for i, location in enumerate(locations):
        image_arr[location[0] - hor_left: location[0] + hor_right, location[1] - ver_left: location[1] + ver_right] = patch

    return image_arr

def convert_ground_truth_to_locations(gd_truths):
    all_idxes = []
    for i, gd_truth in enumerate(gd_truths):
        for j, val in enumerate(gd_truth):
            if val:
                all_idxes.append(i*5 + j)
    return all_idxes

def read_image_as_nparr(img_path):
    img = Image.open(img_path).convert('L')
    if img.mode == "RGBA":
        return np.copy(np.asarray(img)[..., :3])
    else:
        return np.copy(np.asarray(img))

def inject_gd_truth_in_image(img_path, gd_truth_path, target_path):
    img_arr = read_image_as_nparr(img_path)

    patch = get_marker_patch(10, 10)
    locations = get_injection_locations(img_arr.shape[1], img_arr.shape[0])
    gd_truths = read_ground_truths(gd_truth_path)
    idxes = convert_ground_truth_to_locations(gd_truths)
    filtered_locs = filter_location_by_idxes(locations, idxes)
    img_arr = inject_patches_in_image(img_arr, patch, filtered_locs)

    imageio.imsave(target_path, img_arr.astype(np.uint8))

def kernel_to_str(ker):
    ker = ker.reshape(-1)
    return "".join([str(a) + "_" for a in ker])

def all_different_kernels(img_arr, x_vals, y_vals):
    ker_map = {}
    for x,y in zip(x_vals, y_vals):
        ker_map[kernel_to_str(img_arr[x:x+10, y:y+10])] = img_arr[x:x+10, y:y+10]
    return list(ker_map.values())

def is_equal_with_tol(val1, val2, tol = 3):
    return abs(val1 - val2) <= tol

def get_marker_corners(x_vals, y_vals):
    top_pt = [x_vals[0], y_vals[0]]
    bottom_pt = [x_vals[0], y_vals[0]]
    left_pt = [x_vals[0], y_vals[0]]
    right_pt = [x_vals[0], y_vals[0]]

    left_r_top_left = [x_vals[0], y_vals[0]]
    left_r_top_right = [x_vals[0], y_vals[0]]
    left_r_bottom_left = [x_vals[0], y_vals[0]]
    left_r_bottom_right = [x_vals[0], y_vals[0]]

    right_r_top_left = [x_vals[0], y_vals[0]]
    right_r_top_right = [x_vals[0], y_vals[0]]
    right_r_bottom_left = [x_vals[0], y_vals[0]]
    right_r_bottom_right = [x_vals[0], y_vals[0]]

    tol = 3

    for x, y in zip(x_vals, y_vals):
        if x < top_pt[0]:
            top_pt = [x,y]
        if x > bottom_pt[0]:
            bottom_pt = [x,y]
        if y<left_pt[1]:
            left_pt = [x,y]
        if y > right_pt[1]:
            right_pt = [x,y]

        if y <= (left_r_top_left[1]+tol):
            if is_equal_with_tol(y, left_r_top_left[1]) and x < (left_r_top_left[0]+tol):
                left_r_top_left = [x, y]
            elif y < (left_r_top_left[1] - tol):
                left_r_top_left = [x, y]
        if x <= (left_r_top_right[0]+tol):
            if is_equal_with_tol(x, left_r_top_right[0]) and y > (left_r_top_right[1] - tol):
                left_r_top_right = [x, y]
            elif (x < left_r_top_right[0] - tol):
                left_r_top_right = [x, y]
        if x >= (left_r_bottom_left[0]-tol):
            if is_equal_with_tol(x, left_r_bottom_left[0]) and y < (left_r_bottom_left[1] + tol):
                left_r_bottom_left = [x, y]
            elif x > (left_r_bottom_left[0] + tol):
                left_r_bottom_left = [x, y]
        if y >= (left_r_bottom_right[1]-tol):
            if is_equal_with_tol(y, left_r_bottom_right[1]) and x > (left_r_bottom_right[0] - tol):
                left_r_bottom_right = [x, y]
            elif y > (left_r_bottom_right[1] + tol):
                left_r_bottom_right = [x, y]

        if x <= (right_r_top_left[0]+tol):
            if is_equal_with_tol(x, right_r_top_left[0]) and y < (right_r_top_left[1] + tol):
                right_r_top_left = [x, y]
            elif x < (right_r_top_left[0] - tol):
                right_r_top_left = [x, y]
        if y >= (right_r_top_right[1]-tol):
            if is_equal_with_tol(y, right_r_top_right[1]) and x < (right_r_top_right[0] + tol):
                right_r_top_right = [x, y]
            elif y > (right_r_top_right[1] + tol):
                right_r_top_right = [x,y]
        if y <= (right_r_bottom_left[1]+tol):
            if is_equal_with_tol(y, right_r_bottom_left[1]) and x > (right_r_bottom_left[0] - tol):
                right_r_bottom_left = [x, y]
            elif y < (right_r_bottom_left[1] - tol):
                right_r_bottom_left = [x,y]
        if x >= (right_r_bottom_right[0] - tol):
            if is_equal_with_tol(x, right_r_bottom_right[0]) and y > (right_r_bottom_right[1] - tol):
                right_r_bottom_right = [x, y]
            elif x > (right_r_bottom_right[0] + tol):
                right_r_bottom_right = [x,y]

    # Check if right rotation
    if right_r_top_left[1] > right_r_bottom_right[1]:
        return left_r_top_left, left_r_top_right, left_r_bottom_left, left_r_bottom_right
    else:
        return right_r_top_left, right_r_top_right, right_r_bottom_left, right_r_bottom_right

def get_selected_idxes(x_vals, y_vals):
    val_dict = get_value_dict(x_vals, y_vals)
    # top_left = [x_vals[0], y_vals[0]]
    # top_right = [x_vals[0], y_vals[0]]
    # bottom_left = [x_vals[0], y_vals[0]]
    # bottom_right = [x_vals[0], y_vals[0]]

    # for x, y in zip(x_vals, y_vals):
    #     if y <= top_left[1] and x <= top_left[0]:
    #         top_left = [x,y]
    #     if x <= top_right[0] and y >=top_right[1]:
    #         top_right = [x,y]
    #     if x >= bottom_left[0] and y <= bottom_left[1]:
    #         bottom_left = [x,y]
    #     if y>=bottom_right[1] and x >= bottom_right[0]:
    #         bottom_right = [x,y]
    top_left, top_right, bottom_left, bottom_right = get_marker_corners(x_vals, y_vals)

    vert_1 = get_n_points_on_line(top_left, bottom_left, num_points=vertical_injection_count_left + 2)
    vert_2 = get_n_points_on_line(top_right, bottom_right, num_points=vertical_injection_count_right + 2)
    hor_1 = get_n_points_on_line(top_left, top_right, num_points=horizontal_injection_count_up + 2)
    hor_2 = get_n_points_on_line(bottom_left, bottom_right, num_points=horizontal_injection_count_down + 2)

    sel_idxes = [[False for j in range(5) ] for i in range(85)]

    #v2 h2 v1 h1
    for i, pt in enumerate(vert_2[1:-1] + hor_2[1:-1] + vert_1[1:-1] + hor_1[1:-1]):
        is_marked = check_proximity(val_dict, pt[0], pt[1], val_range=3, exlusive_search=False)
        if is_marked:
            sel_idxes[int(i/5)][i - int(i/5) * 5] = True

    return sel_idxes

def convert_gd_truth_to_text(gd_truths):
    let_map = {i:a for i, a in enumerate("ABCDE")}
    lines = []
    for i, gd_truth in enumerate(gd_truths):
        line = str(i+1) + " "
        for j, val in enumerate(gd_truth):
            if val:
                line += let_map[j]
        lines.append(line)
    return "\n".join(lines)

def write_to_file(lines, filename):
    with open(filename, "w") as f:
        f.write(lines)

def extract_injected_markers(injected_path, extract_path):
    img_arr = read_image_as_nparr(injected_path)
    ker = get_marker_patch(10, 10, detect=True)
    kernel_act = apply_kernel_all_channels(img_arr, ker)
    # val, indices = torch.topk(torch.from_numpy(kernel_act).view(-1), 429, sorted=True)
    indices = np.argpartition(kernel_act.reshape(-1), -429)[-429:]
    indices = indices[np.argsort(-kernel_act.reshape(-1)[indices])]

    x_vals, y_vals = get_x_y_from_flatten_arr(indices, kernel_act.shape)

    # ker_map = all_different_kernels(img_arr, x_vals, y_vals)
    sel_idxes = get_selected_idxes(x_vals, y_vals)
    lines = convert_gd_truth_to_text(sel_idxes)
    write_to_file(lines, extract_path)
    return sel_idxes

# gd_truth_path = "test-images/c-33_groundtruth.txt"
# gd_truths = read_ground_truths(gd_truth_path)
# inject_gd_truth_in_image("test-images/blank_form.jpg", gd_truth_path, "test-images/injected.png")
#
# print("Injected...")
#
# read_gd_truths = extract_injected_markers("test-images/injected.png", "test-images/extracted.txt")
#
# if (np.array(gd_truths) != np.array(read_gd_truths)).sum() != 0:
#     print("ERROR")
# else:
#     print("Parsed correctly")
