import cv2
import numpy as np
import os

from utils.Functions import getConnectedComponents, getSkeletonOfImage, resizeImages, getSingleMaxBoundingBoxOfImage, \
    createBlankGrayscaleImage


def is_comp_contain_stroke(component, stroke):
    """
     check wheter this component contains this stroke
    :param component:
    :param stroke:
    :return:
    """
    total_num = vaild_num = 0
    for i in range(stroke.shape[0]):
        for j in range(stroke.shape[1]):
            if stroke[i][j] == 255:
                continue
            if stroke[i][j] == 0:
                if component[i][j] == 0:
                    vaild_num += 1
                total_num += 1

    if vaild_num * 1. / total_num >= 0.98:
        return True
    return False


def is_same_components(src_comp, temp_comp):
    """
    Check two components are correlations via calculating coverage area > 75%
    :param src_comp:
    :param temp_comp:
    :return:
    """
    if src_comp.shape != temp_comp.shape:
        print("two components are not same shape!")
        return False
    total_pixels = valid_pixels = 0
    for i in range(src_comp.shape[0]):
        for j in range(src_comp.shape[1]):
            if src_comp[i][j] == 0.:
                if temp_comp[i][j] == 0.:
                    valid_pixels += 1
                total_pixels += 1
    # print(1. * valid_pixels / total_pixels)

    if 1. * valid_pixels / total_pixels > .4:
        # two components are same
        return True
    return False


path = "../test_images/gong_copy.png"

temp_path = "../test_images/"
temp_char_path = "../test_images/page1_char_3.png"

# load template strokes
temp_strokes = []
for i in range(1, 10):
    temp_img_path = os.path.join(temp_path, ("stroke_%d.png" % i))
    temp_img = cv2.imread(temp_img_path, 0)
    temp_strokes.append(temp_img)
print("temp stroke num: ", len(temp_strokes))

# template char image
temp_char_img = cv2.imread(temp_char_path, 0)

# load gong copy image
src_img = cv2.imread(path, 0)

# resize src image and template image
temp_minx, temp_miny, temp_minw, temp_minh = getSingleMaxBoundingBoxOfImage(temp_char_img)
src_minx, src_miny, src_minw, src_minh = getSingleMaxBoundingBoxOfImage(src_img)

temp_box_width = max(temp_minw, temp_minh)
src_box_width = max(src_minw, src_minh)

print("tempbox w:", temp_box_width, "src box w:", src_box_width)

temp_img_box = temp_char_img[temp_miny: temp_miny + temp_minh, temp_minx: temp_minx + temp_minw]
src_img_box = src_img[src_miny: src_miny + src_minh, src_minx: src_minx + src_minw]
print(temp_img_box.shape)
print(src_img_box.shape)

# create square of temp and src image
temp_square = np.array(np.ones((temp_box_width, temp_box_width)) * 255, dtype=np.uint8)
src_square = np.array(np.ones((src_box_width, src_box_width)) * 255, dtype=np.uint8)

# paste temp and src image to square image
if temp_minw > temp_minh:
    temp_square[int((temp_minw-temp_minh)/2): int((temp_minw-temp_minh)/2)+temp_minh, 0:temp_minw] = temp_img_box
elif temp_minw < temp_minh:
    temp_square[0: temp_minh, int((temp_minh-temp_minw)/2): int((temp_minh-temp_minw)/2)+temp_minw] = temp_img_box

if src_minw > src_minh:
    src_square[int((src_minw - src_minh) / 2): int((src_minw - src_minh)/2) + src_minh, 0:src_minw] = src_img_box
elif src_minw < src_minh:
    src_square[0: src_minh, int((src_minh - src_minw) / 2): int((src_minh - src_minw) / 2) + src_minw] = src_img_box

# scale template and src image to become same size, the size is smaller one
scale_ratio = 1. * min(temp_box_width, src_box_width) / max(temp_box_width, src_box_width)
print("scale ratio: ", scale_ratio)

if temp_box_width > src_box_width:
    temp_square = cv2.resize(temp_square, (src_square.shape[0], src_square.shape[1]))
elif temp_box_width < src_box_width:
    src_square = cv2.resize(src_square, (temp_square.shape[0], temp_square.shape[1]))


# resize template
temp_strokes_resized = []
for i in range(len(temp_strokes)):
    orig_stroke = temp_strokes[i]

    stroke_box = orig_stroke[temp_miny: temp_miny+temp_minh, temp_minx: temp_minx+temp_minw]

    stroke_square = np.ones((temp_box_width, temp_box_width)) * 255

    if temp_minw > temp_minh:
        stroke_square[int((temp_minw - temp_minh) / 2): int((temp_minw - temp_minh) / 2) + temp_minh,
        0:temp_minw] = stroke_box
    elif temp_minw < temp_minh:
        stroke_square[0: temp_minh,
        int((temp_minh - temp_minw) / 2): int((temp_minh - temp_minw) / 2) + temp_minw] = stroke_box

    if temp_box_width > src_box_width:
        stroke_square = cv2.resize(stroke_square, (src_square.shape[0], src_square.shape[1]))
    temp_strokes_resized.append(stroke_square)


# detect components in src image
src_components = getConnectedComponents(src_square, connectivity=8)
print("src components num: ", len(src_components))

# detect components in template image
temp_components = getConnectedComponents(temp_square)
print("template components num: ", len(temp_components))

if len(temp_components) == len(src_components):
    print("components are same!")
else:
    print("components are not same!!!")

# detect template char components and strokes contain relation
temp_comp_contains_strokes = []
for i in range(len(temp_components)):
    strokes = []
    for j in range(len(temp_strokes_resized)):
        if is_comp_contain_stroke(temp_components[i], temp_strokes_resized[j]):
            strokes.append(j)

    temp_comp_contains_strokes.append(strokes)
print("comp stroke relation: ", temp_comp_contains_strokes)

# detect src and template components relation
temp_src_comp_relations = []

for i in range(len(src_components)):
    src = src_components[i]

    for j in range(len(temp_components)):
        temp = temp_components[j]

        if is_same_components(src, temp):
            temp_src_comp_relations.append((i, j))
print("temp src compoents relation: ", temp_src_comp_relations)

# infer src components strokes relations
src_comp_contains_strokes = []
for i in range(len(temp_src_comp_relations)):
    src_id, temp_id = temp_src_comp_relations[i]
    strokes = temp_comp_contains_strokes[i]
    src_comp_contains_strokes.append(strokes)
print("src comp contains strokes: ", src_comp_contains_strokes)


# stroke extraction
basic_strokes = []
processed_src_comps = []
for i in range(len(src_comp_contains_strokes)):
    if len(src_comp_contains_strokes[i]) == 1:
        # component is a single stroke
        basic_strokes.append(src_components[i])
    else:
        # component contains several strokes
        pass

print("basic strokes num: ", len(basic_strokes))

# for i in range(len(basic_strokes)):
#     cv2.imshow("stroke_%d" % i, basic_strokes[i])



cv2.imshow("src", src_img)
cv2.imshow("temp char", temp_char_img)
#
# cv2.imshow("temp square", temp_square)
# cv2.imshow("src square", src_square)

# for i in range(len(temp_strokes_resized)):
#     stroke = temp_strokes_resized[i]
#     cv2.imshow("stroke%d" % i, stroke)

# for i in range(len(src_components)):
#     cv2.imwrite("src_comp_%d.png" % i, src_components[i])
#
# for i in range(len(temp_strokes_resized)):
#     cv2.imwrite("temp_stroke_%d.png" % i, temp_strokes_resized[i])


cv2.waitKey(0)
cv2.destroyAllWindows()











#
# _, orig_thresh = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
# bin_thresh = (orig_thresh == 0).astype(int)
#
# skeleton_1 = getSkeletonOfImage(char_img)
#
#
# # steps one and two, condition one
# def pixel_is_black(arr, x, y):
#     return arr[x, y] == 1
#
#
# # steps one and two, condition two
# def pixel_has_2_to_6_black_neighbors(arr, x, y):
#     if (2 <= arr[x, y-1] + arr[x+1, y-1] + arr[x+1, y] + arr[x+1, y+1] +
#         arr[x, y+1] + arr[x-1, y+1] + arr[x-1, y] + arr[x-1, y-1] <= 6):
#         return True
#     return False
#
# # steps one and two, condition three
# def pixel_has_1_white_to_black_neighbor_transition(arr, x, y):
#     # neighbor is a list of neighbor pixel values: neighbor P2 appears
#     # twice since we will cycle around P1
#     neighbors = [arr[x, y-1], arr[x+1, y-1], arr[x+1, y], arr[x+1, y+1],
#                  arr[x, y+1], arr[x, y+1], arr[x-1, y], arr[x-1, y-x],
#                  arr[x, y-1]]
#
#     # zip returns iterator of tuples composed of a neighbor and next neighbor
#     # we then check if the neighbor and next neighbor is a 0 -> 1 transition
#     # finally, we sum the transitions and return True if there is only one
#     transitions = sum((a, b) == (0, 1) for a, b in zip(neighbors, neighbors[1:]))
#     return transitions == 1
#
# # step one condition four
# def at_least_one_of_P2_P4_P6_is_white(arr, x, y):
#     # if at least of P2 p4 P6 is 0, logit statement will evaluate to false
#     if (arr[x, y-1] and arr[x+1, y] and arr[x, y+1]) == False:
#         return True
#     return False
#
# # step one condition five
# def at_least_one_of_P4_p6_p8_is_white(arr, x, y):
#     # if at least of p4, p6, p8 is white, logic statement will evaluate to false.
#     if (arr[x+1, y] and arr[x, y+1] and arr[x-1, y] == False):
#         return True
#     return False
#
# # step two condition four
# def at_least_one_of_P2_P4_P8_is_white(arr, x, y):
#     # if at least of p2, p4, p8 is white, logic statement will evaluate to false.
#     if (arr[x, y-1] and arr[x+1, y] and arr[x-1, y]) == False:
#         return True
#     return False
#
# # step two condition five
# def at_least_one_of_P2_p6_p8_is_white(arr, x, y):
#     if (arr[x, y-1] and arr[x, y+1] and arr[x-1, y]) == False:
#         return True
#     return False
#
# thinned_thresh = bin_thresh.copy()
#
#
# while True:
#     thresh_copy = thinned_thresh.copy()
#     pixels_meeting_criteria = []
#
#     for i in range(1, thinned_thresh.shape[0] - 1):
#         for j in range(1, thinned_thresh.shape[1] - 1):
#             if (pixel_is_black(thinned_thresh, i, j) and
#             pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
#             pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
#             at_least_one_of_P2_P4_P6_is_white(thinned_thresh, i, j) and
#             at_least_one_of_P4_p6_p8_is_white(thinned_thresh, i, j)):
#                 pixels_meeting_criteria.append((i, j))
#
#     # change noted pixels in tinned threshold array to 0
#     for pixel in pixels_meeting_criteria:
#         thinned_thresh[pixel] = 0
#
#     # step two
#     for i in range(1, thinned_thresh.shape[0] - 1):
#         for j in range(1, thinned_thresh.shape[1] - 1):
#             if (pixel_is_black(thinned_thresh, i, j) and
#             pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
#             pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
#             at_least_one_of_P2_P4_P8_is_white(thinned_thresh, i, j) and
#             at_least_one_of_P2_p6_p8_is_white(thinned_thresh, i, j)):
#                 pixels_meeting_criteria.append((i, j))
#     for pixel in pixels_meeting_criteria:
#         thinned_thresh[pixel] = 0
#
#     if np.all(thresh_copy == thinned_thresh) == True:
#         break
#
#
# thresh = (thinned_thresh == 0).astype(np.uint8)
#
# thresh *= 255
