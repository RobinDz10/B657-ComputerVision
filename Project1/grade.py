import copy
import math
import sys
from PIL import Image
import numpy as np

option_box_height = 35
option_box_width = 33
inter_option_box_width = 60
inter_option_box_height = 48
question_error_threshold_width_avg = 3
question_error_threshold_pt = 2
selection_threshold = -170000
custom_text_width = 60
custom_ans_act_limit = 514000
custom_text_dist_option = 120
option_idx = {l: i for i, l in enumerate("ABCDE")}


def get_gauss_dist(x_val, sigma):
    return math.exp(-((x_val / sigma) ** 2) / 2) / sigma


def detail_kernel(size, sigma):
    if size % 2 == 0:
        raise NotImplementedError("Size should be odd.")
    g_kern = create_gaussian_kernel(size, sigma)
    id_kernel = np.zeros((size, size))
    id_kernel[int(size / 2), int(size / 2)] = 1
    dt_kernel = id_kernel - g_kern
    return dt_kernel


def sharpen_kernel(size, sigma):
    dt_kernel = detail_kernel(size, sigma)
    id_kernel = np.zeros((size, size))
    id_kernel[int(size / 2), int(size / 2)] = 1
    sp_kernel = dt_kernel + id_kernel
    return sp_kernel


def get_option_kernel():
    ker = np.zeros((option_box_height, option_box_width))
    ker[:2, :] = -1
    ker[:, :2] = -1
    ker[:, -2:] = -1
    ker[-2:, :] = -1
    return ker


def create_gaussian_kernel(size, sigma):
    ker = np.zeros((size, size))
    for i in range(size):
        x_idx = int(abs(size / 2 + 0.5 - i - 1))
        val_x = get_gauss_dist(x_idx, sigma)
        for j in range(size):
            y_idx = int(abs(size / 2 + 0.5 - j - 1))
            val_y = get_gauss_dist(y_idx, sigma)
            ker[i, j] = (val_x ** 2 + val_y ** 2) ** 0.5

    return ker / np.sum(ker)


def apply_kernel_all_channels(img_arr, kernel, channel_first=True):
    if len(img_arr.shape) == 2:
        return apply_kernel(img_arr, kernel)
    channel_dim = 0 if channel_first else -1
    result = None  # np.expand_dims(np.zeros((img_arr.shape[0 + channel_first]+1-kernel.shape[0] , img_arr.shape[1+channel_first]+1-kernel.shape[1])), axis = channel_dim)
    for i in range(img_arr.shape[0] if channel_first else img_arr.shape[-1]):
        channel_arr = img_arr[i] if channel_first else img_arr[..., i]
        if i != 0:
            result = np.concatenate((result, np.expand_dims(apply_kernel(channel_arr, kernel), axis=channel_dim)),
                                    axis=channel_dim)
        else:
            result = np.expand_dims(apply_kernel(channel_arr, kernel), axis=channel_dim)
    return result


def apply_kernel(img_arr, kernel):
    result = np.zeros((img_arr.shape[0] + 1 - kernel.shape[0], img_arr.shape[1] + 1 - kernel.shape[1]))
    for i in range(img_arr.shape[0] + 1 - kernel.shape[0]):
        for j in range(img_arr.shape[1] + 1 - kernel.shape[1]):
            result[i, j] = np.sum(img_arr[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    return result


def get_x_y_from_flatten_arr(idxes, shap):
    x_vals = (idxes / shap[1]).astype(int)
    y_vals = idxes - x_vals * shap[1]
    return x_vals, y_vals


def check_proximity(val_dict, x, y, val_range, exlusive_search=True):
    for i in range(x - val_range, x + val_range + 1):
        for j in range(y - val_range, y + val_range + 1):
            if exlusive_search and i == x and j == y:
                continue
            if i in val_dict and j in val_dict[i]:
                return True
    return False


def get_value_dict(x_vals, y_vals):
    val_dict = {}
    for x, y in zip(x_vals, y_vals):
        if x not in val_dict:
            val_dict[x] = {}
        if y not in val_dict[x]:
            val_dict[x][y] = 1
    return val_dict


def get_all_points_horizontal(x, val_dict):
    all_points = []
    for x in range(x, x + 10):
        if x in val_dict:
            for y in val_dict[x]:
                all_points.append((x, y))
    return all_points


def is_point_a_question(x, y, all_points):
    total_pt_fount = 0
    total_width_error = 0
    y_width = inter_option_box_width
    option_points = [(x, y, True)]

    pre_pt = y + y_width
    for i in range(4):
        # Find a point, if not found, create an imaginary point.
        pt_found = False
        for point in all_points:
            if point[1] >= (pre_pt - 4) and point[1] <= (pre_pt + 4):
                total_width_error += abs(point[1] - pre_pt)
                pre_pt = point[1] + y_width
                total_pt_fount += 1
                pt_found = True
                option_points.append((point[0], point[1], True))
                break
        if not pt_found:
            option_points.append((x, pre_pt, False))
            pre_pt += y_width
    return ((total_pt_fount >= question_error_threshold_pt) and (
                (total_width_error / total_pt_fount) <= question_error_threshold_width_avg)), \
           option_points


# Returns top line and error
def get_top_line(x_vals, y_vals, x_thresh=0):
    val_dict = get_value_dict(x_vals, y_vals)
    min_x = min(x_vals)

    all_points = []
    for x in range(min_x, min_x + 5):
        if x in val_dict:
            for y in val_dict[x]:
                all_points.append((x, y))

    x_err = {x: 0 for x in range(min_x, min_x + 5)}
    for x in range(min_x, min_x + 5):
        for point in all_points:
            x_err[x] += (point[0] - x) ** 2
    min_x_err = min_x
    for x in x_err:
        if x_err[x] < x_err[min_x_err]:
            min_x_err = x


def get_top_not_in_proximity(x_vals, y_vals, k, proximity=10):
    val_dict = get_value_dict(x_vals, y_vals)

    removed_x, removed_y = [], []
    filter_x, filter_y = [], []
    for x, y in zip(x_vals, y_vals):
        if check_proximity(val_dict, x, y, proximity):
            removed_x.append(x)
            removed_y.append(y)
            del val_dict[x][y]
            continue
        filter_x.append(x)
        filter_y.append(y)

        if len(filter_y) == k:
            break
    return np.array(filter_x), np.array(filter_y), np.array(removed_x), np.array(removed_y)


def is_closer(x_vals, y_vals):
    val_dict = get_value_dict(x_vals, y_vals)
    total_proximities = 0
    for x, y in zip(x_vals, y_vals):
        if check_proximity(val_dict, x, y, 3):
            total_proximities += 1
    print(f"Total Proximities: {total_proximities}")


def delete_val_dict_pair(val_dict, x, y):
    del val_dict[x][y]
    if len(val_dict[x]) == 0:
        del val_dict[x]


def init_group(options):
    return {
        "all_options": [options],
        "last_x": options[0][0],
    }


def check_if_belongs_to_group(option_list, group_list):
    for group in group_list:
        next_x = group["last_x"] + inter_option_box_height
        x_match, y_match = False, False
        for option in option_list:  # if any option overlaps y position
            if (option[0] > (next_x - 5)) and (option[0] < (next_x + 5)):
                x_match = True
                break
        if not x_match:
            continue

        # check if any y option overlaps
        need_adjustment = True
        for j, option in enumerate(option_list):
            for i, prev_option in enumerate(group["all_options"][-1]):
                if (option[1] < (prev_option[1] + 5)) and (option[1] > (prev_option[1] - 5)):
                    y_match = True
                    if i == 0 and j == 0:
                        need_adjustment = False
                    break
            if y_match:
                break
        if y_match:
            return True, group, need_adjustment
    return False, None, None


def shift_options(options, count):
    new_options = options[:len(options) - count]
    for i in range(count):
        x = int(2 * new_options[0][0] - new_options[1][0])
        y = int(2 * new_options[0][1] - new_options[1][1])
        new_options = [(x, y, False)] + new_options
    return new_options


def readjust_group(group, new_options):
    prev_options = group["all_options"][-1]
    if prev_options[0][1] < new_options[0][1]:
        # only need to adjust the current
        count = 0
        gp_ops_last = group["all_options"][-1]
        for option in gp_ops_last:
            if (new_options[0][1] > (option[1] - 5)) and (new_options[0][1] < (option[1] + 5)):
                break
            count += 1
        shifted_options = shift_options(new_options, count)
        return group, new_options, shifted_options
    else:
        # adjust all previous
        # find the count of shifts
        count = 0
        gp_ops_last = group["all_options"][-1][0]
        for option in new_options:
            if (gp_ops_last[1] > (option[1] - 5)) and (gp_ops_last[1] < (option[1] + 5)):
                break
            count += 1

        new_all_options = []
        for option_list in group["all_options"]:
            shifted_options = shift_options(option_list, count)
            new_all_options.append(shifted_options)
        group["all_options"] = new_all_options
        group["last_x"] = new_all_options[-1][0][0]
        return group, new_options, new_options


def add_to_group(group, options):
    group["all_options"].append(options)
    group["last_x"] = options[0][0]


def group_questions(q_begs):
    vertical_groups = []

    for x in sorted(q_begs.keys()):
        for y in sorted(q_begs[x].keys()):
            belongs, group, need_adj = check_if_belongs_to_group(q_begs[x][y], vertical_groups)
            if belongs:
                if need_adj:
                    g, temp1, new_options = readjust_group(group, q_begs[x][y])
                    add_to_group(group, new_options)
                else:
                    add_to_group(group, q_begs[x][y])
            else:
                new_group = init_group(q_begs[x][y])
                vertical_groups.append(new_group)

    return vertical_groups


def are_group_combineable(gp1, gp2):
    first_options = gp2["all_options"][0]
    last_options = gp1["all_options"][-1]

    tolerance_x = gp1["last_x"] + 2.2 * inter_option_box_height
    if tolerance_x < first_options[0][0] or last_options[0][0] > first_options[0][0]:
        return False, 0, False

    y11, y12, y21, y22 = first_options[0][1], first_options[-1][1], last_options[0][1], last_options[1][1]

    if not (y12 > y21 and y22 > y11):
        return False, 0, False

    if (y11 > (y21 - 5)) and (y11 < (y21 + 5)):
        return True, round((first_options[0][0] - last_options[0][0]) / inter_option_box_height) - 1, False
    else:
        return True, round((first_options[0][0] - last_options[0][0]) / inter_option_box_height) - 1, True


def create_imag_rows(opt1, opt2, num_rows):  # opt2 > opt1
    opts = []
    for i in range(num_rows):
        opt = []
        for j in range(len(opt1)):
            x = round(opt1[j][0] + (opt2[j][0] - opt1[j][0]) / (num_rows + 1))
            y = round(opt1[j][1] + (opt2[j][1] - opt1[j][1]) / (num_rows + 1))
            opt.append((x, y, False))
        opts.append(opt)
    return opts


def merge_groups(gp1, gp2, num_rows):
    img_rows = create_imag_rows(gp1["all_options"][-1], gp2["all_options"][0], num_rows)
    gp1["all_options"].extend(img_rows)
    gp1["all_options"].extend(gp2["all_options"])
    gp1["last_x"] = gp2["last_x"]


def combine_groups(vertical_groups):
    for i, gp1 in enumerate(vertical_groups):
        for j, gp2 in enumerate(vertical_groups):
            if (i == j) or "invalid" in gp1 or "invalid" in gp2:
                continue
            should_combine, mis_row, need_adj = are_group_combineable(gp1, gp2)
            if should_combine:
                print(f"Combining...{i}_{j}")
                gp2["invalid"] = True
                if need_adj:
                    print(f"adjusting...{i}_{j}")
                    if gp1["all_options"][-1][0][1] > gp2["all_options"][0][0][1]:
                        temp0, temp1, temp2 = readjust_group(gp1, gp2["all_options"][0])
                    else:
                        temp0, temp1, temp2 = readjust_group(gp2, gp1["all_options"][-1])
                merge_groups(gp1, gp2, mis_row)
    return [gp for gp in vertical_groups if "invalid" not in gp]


# Takes in non-max suppressed x and y values
def find_all_questions_beginings(x_vals, y_vals):
    val_dict = get_value_dict(x_vals, y_vals)
    running_val_dict = copy.deepcopy(val_dict)
    total_rem_points = len(x_vals)
    question_beg = {}  # x:[y]

    while total_rem_points > 0:
        min_x = min(running_val_dict.keys())
        all_points_min_x = get_all_points_horizontal(min_x, running_val_dict)
        min_y = min([pt[1] for pt in all_points_min_x])
        min_x = min([pt[0] for pt in all_points_min_x if pt[1] == min_y])

        is_q, option_points = is_point_a_question(min_x, min_y, all_points_min_x)
        if is_q:
            if min_x not in question_beg:
                question_beg[min_x] = {}
            if min_y not in question_beg[min_x]:
                question_beg[min_x][min_y] = option_points

            # Remove all points
            for pt in option_points:
                if pt[2]:
                    total_rem_points -= 1
                    delete_val_dict_pair(running_val_dict, *pt[:2])
        else:
            total_rem_points -= 1
            delete_val_dict_pair(running_val_dict, min_x, min_y)
    return question_beg


def get_selected_option_kernel():
    return np.full((option_box_height, option_box_width), -1)


def get_selected_options(img_arr, option_list):
    ker = get_selected_option_kernel()
    kernel_vals = []
    for option in option_list:
        kernel_vals.append(np.sum(img_arr[option[0]:option[0] + option_box_height, option[1]:option[1] + option_box_width] * ker))
    return [val > selection_threshold for val in kernel_vals], kernel_vals


def read_ground_truths(filename, st_answer=False):
    gd_truth = [[False for i in range(5)] for i in range(85)]
    custom_answer_idxes = []
    with open(filename, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        ans = line.split()
        op_list = [False] * 5
        if len(ans) > 1:
            for op in ans[1]:
                op_list[option_idx[op]] = True
        if st_answer and len(ans) == 3 and ans[2] == "x":
            custom_answer_idxes.append(i)

        gd_truth[int(ans[0]) - 1] = op_list
    if st_answer:
        return gd_truth, custom_answer_idxes
    return gd_truth


def perform_selected_option_analytics(checked_options, gd_truths, q_markers):
    sel_min_max = [[50 * 50 * 255, -50 * 50 * 255] for i in range(5)]
    un_sel_min_max = [[50 * 50 * 255, -50 * 50 * 255] for i in range(5)]
    sel_min_max_idx = [[[0, 0], [0, 0]] for i in range(5)]
    un_sel_min_max_idx = [[[0, 0], [0, 0]] for i in range(5)]

    for ch_op, gd_truth, q_marker in zip(checked_options, gd_truths, q_markers):
        for i, (op_val, sel, op_mark) in enumerate(zip(ch_op, gd_truth, q_marker)):
            if sel:
                if op_val < sel_min_max[i][0]:
                    sel_min_max[i][0] = op_val
                    sel_min_max_idx[i][0] = op_mark[:2]
                if op_val > sel_min_max[i][1]:
                    sel_min_max[i][1] = op_val
                    sel_min_max_idx[i][1] = op_mark[:2]
            else:
                if op_val < un_sel_min_max[i][0]:
                    un_sel_min_max[i][0] = op_val
                    un_sel_min_max_idx[i][0] = op_mark[:2]
                if op_val > un_sel_min_max[i][1]:
                    un_sel_min_max[i][1] = op_val
                    un_sel_min_max_idx[i][1] = op_mark[:2]
    return sel_min_max, \
           un_sel_min_max, \
           min([op[0] for op in sel_min_max]), \
           max([op[1] for op in sel_min_max]), \
           min([op[0] for op in un_sel_min_max]), \
           max([op[1] for op in un_sel_min_max]), \
           sel_min_max_idx, \
           un_sel_min_max_idx

def does_q_match_horizontally(ops_0, ops_1):
    y_change_intra = ops_0[-1][1] - ops_0[0][1]
    y_change_inter = ops_1[0][1] - ops_0[0][1]
    x_change_intra = ops_0[-1][0] - ops_0[0][0]
    x_change_inter = y_change_inter * x_change_intra / y_change_intra
    x_val = ops_0[0][0] + x_change_inter
    return abs(x_val - ops_1[0][0]) < 5

def create_next_row(row_1, row_2):
    next_row = []
    for o1, o2 in zip(row_1, row_2):
        next_row.append([o2[0] * 2 - o1[0], o2[1] * 2 - o1[1]])
    return next_row

def add_beg_end_questions_if_needed(vert_groups): # In left-right order
    if len(vert_groups[0]["all_options"]) < 29:
        if does_q_match_horizontally(vert_groups[0]["all_options"][0], vert_groups[1]["all_options"][0]):
            if does_q_match_horizontally(vert_groups[0]["all_options"][-1], vert_groups[1]["all_options"][-1]):
                if len(vert_groups[1]["all_options"]) < 29:
                    if does_q_match_horizontally(vert_groups[0]["all_options"][0], vert_groups[2]["all_options"][0]):
                        vert_groups[0]["all_options"] = [create_next_row(vert_groups[0]["all_options"][1], vert_groups[0]["all_options"][0])] + vert_groups[0]["all_options"]
                        vert_groups[1]["all_options"] = [create_next_row(vert_groups[1]["all_options"][1], vert_groups[1]["all_options"][0])] + vert_groups[1]["all_options"]
                        #Add row on top of both 0 and 1
                else:
                    # Add row at the bottom of both 0 and 1
                    vert_groups[0]["all_options"].append(create_next_row(vert_groups[0]["all_options"][-2],vert_groups[0]["all_options"][-1]))
                    vert_groups[1]["all_options"].append(create_next_row(vert_groups[1]["all_options"][-2],vert_groups[1]["all_options"][-1]))
            else:
                # Add row at the bottom of 0
                vert_groups[0]["all_options"].append(create_next_row(vert_groups[0]["all_options"][-2], vert_groups[0]["all_options"][-1]))
                pass
        else:
            # Add row at the top of 0
            vert_groups[0]["all_options"] = [create_next_row(vert_groups[0]["all_options"][1],vert_groups[0]["all_options"][0])] + vert_groups[0]["all_options"]
    if len(vert_groups[1]["all_options"]) < 29:
        if does_q_match_horizontally(vert_groups[0]["all_options"][0], vert_groups[1]["all_options"][0]):
            if does_q_match_horizontally(vert_groups[0]["all_options"][-1], vert_groups[1]["all_options"][-1]):
                if len(vert_groups[1]["all_options"]) < 29:
                    if does_q_match_horizontally(vert_groups[0]["all_options"][0], vert_groups[2]["all_options"][0]):
                        vert_groups[0]["all_options"] = [create_next_row(vert_groups[0]["all_options"][1],vert_groups[0]["all_options"][0])] + vert_groups[0]["all_options"]
                        vert_groups[1]["all_options"] = [create_next_row(vert_groups[1]["all_options"][1], vert_groups[1]["all_options"][0])] + vert_groups[1]["all_options"]
                        #Add row on top of both 0 and 1
                else:
                    vert_groups[0]["all_options"].append(create_next_row(vert_groups[0]["all_options"][-2],vert_groups[0]["all_options"][-1]))
                    vert_groups[1]["all_options"].append(create_next_row(vert_groups[1]["all_options"][-2],vert_groups[1]["all_options"][-1]))
                    # Add row at the bottom of both 0 and 1
                    pass
            else:
                vert_groups[1]["all_options"].append(create_next_row(vert_groups[1]["all_options"][-2], vert_groups[1]["all_options"][-1]))
                # Add row at the bottom of 1
        else:
            vert_groups[1]["all_options"] = [create_next_row(vert_groups[1]["all_options"][1],vert_groups[1]["all_options"][0])] + vert_groups[1]["all_options"]
            # Add row at the top of 1

    if len(vert_groups[2]["all_options"]) < 27:
        if does_q_match_horizontally(vert_groups[2]["all_options"][0], vert_groups[1]["all_options"][0]):
            vert_groups[2]["all_options"].append(create_next_row(vert_groups[2]["all_options"][-2], vert_groups[2]["all_options"][-1]))
            # add 1 at the bottom of 2
        else:
            vert_groups[2]["all_options"] = [create_next_row(vert_groups[2]["all_options"][1],vert_groups[2]["all_options"][0])] + vert_groups[2]["all_options"]
            # Add 1 at the top of 2

def get_question_marker_arr(groups):
    filtered_gps = []
    for i in range(len(groups)):
        if len(groups[i]["all_options"]) < 2:
            continue
        filtered_gps.append(groups[i])
    y_vals = [int(gp["all_options"][0][0][1]) for gp in filtered_gps]
    sorted_y = sorted(y_vals)

    # s_v_gps = [filtered_gps[y_vals.index(sy)] for sy in sorted_y]
    # s_v_gps[1]["all_options"] = s_v_gps[1]["all_options"][1:]
    # s_v_gps[0]["all_options"] = s_v_gps[0]["all_options"][:-1]

    add_beg_end_questions_if_needed([filtered_gps[y_vals.index(sy)] for sy in sorted_y])
    marker_arr = []
    for y_val in sorted_y:
        idx = y_vals.index(y_val)
        sel_group = filtered_gps[idx]
        for op_list in sel_group["all_options"]:
            op_markers = []
            for op in op_list:
                op_markers.append(op[:2])
            marker_arr.append(op_markers)
    return marker_arr

def get_sel_q_markers(img_arr, q_markers):
    q_ker_vals = []
    sel_lists = []
    for q_marker in q_markers:
        sel_list, ker_vals = get_selected_options(img_arr, q_marker)
        q_ker_vals.append(ker_vals)
        sel_lists.append(sel_list)
    return q_ker_vals, sel_lists


def get_hand_writing_kernel():
    return np.ones((option_box_height, custom_text_width))

def check_if_custom_answer(q_marker, img_arr):
    img_patch = img_arr[q_marker[0][0]: q_marker[0][0] + option_box_height,
                q_marker[0][1] - custom_text_dist_option: q_marker[0][1] + custom_text_width - custom_text_dist_option]
    ker = get_hand_writing_kernel()
    act_val = (img_patch * ker).sum()
    return act_val <= custom_ans_act_limit, act_val

def find_all_custom_answers(img_arr, q_markers, gd_truths):
    custom_answer_idxes = []
    act_vals = []
    for i, gd_truth in enumerate(gd_truths):
        is_custom, act_val = check_if_custom_answer(q_markers[i], img_arr)
        act_vals.append(act_val)
        # if sum(gd_truth) > 1:
        if is_custom:
            custom_answer_idxes.append(i)
    return custom_answer_idxes, act_vals

def read_image_as_nparr(img_path):
    img = Image.open(img_path).convert('L')
    if img.mode == "RGBA":
        return np.copy(np.asarray(img)[..., :3])
    else:
        return np.copy(np.asarray(img))

def convert_gd_truth_to_text(gd_truths, all_custom_answers):
    let_map = {i:a for i, a in enumerate("ABCDE")}
    lines = []
    for i, gd_truth in enumerate(gd_truths):
        line = str(i+1) + " "
        for j, val in enumerate(gd_truth):
            if val:
                line += let_map[j]
        if i in all_custom_answers:
            line += " x"
        lines.append(line)
    return "\n".join(lines)

def write_to_file(lines, filename):
    with open(filename, "w") as f:
        f.write(lines)

def grade_input_image(img_path, output_path):
    img_arr = read_image_as_nparr(img_path)
    kernel = get_option_kernel()
    det_img = apply_kernel_all_channels(img_arr, kernel, False)

    # vals, idxes = torch.topk(torch.from_numpy(det_img).view(-1), 1000, sorted=True)
    idxes = np.argpartition(det_img.reshape(-1), -1000)[-1000:]
    idxes = idxes[np.argsort(-det_img.reshape(-1)[idxes])]

    x_vals, y_vals = get_x_y_from_flatten_arr(idxes, det_img.shape)
    x_vals, y_vals, rem_x, rem_y = get_top_not_in_proximity(x_vals, y_vals, 435)
    x_vals = [int(x) for x in x_vals]
    y_vals = [int(y) for y in y_vals]

    q_beg = find_all_questions_beginings(x_vals, y_vals)

    all_vertical_groups = group_questions(q_beg)
    all_vertical_groups = combine_groups(all_vertical_groups)

    q_markers = get_question_marker_arr(all_vertical_groups)
    all_kernel_vals, all_sel_vals = get_sel_q_markers(img_arr, q_markers)
    custom_ans_idxes, act_vals = find_all_custom_answers(img_arr, q_markers, all_sel_vals)

    lines = convert_gd_truth_to_text(all_sel_vals, custom_ans_idxes)
    write_to_file(lines, output_path)

if __name__ == '__main__':
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    grade_input_image(img_path, output_path)
