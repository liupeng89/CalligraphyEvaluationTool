# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# from functions.AddBoundingBox import addBoundingBox
#
# src_path = "../characters/src_dan_processed.png"
# tag_path = "../characters/tag_dan_processed.png"
#
# src_img = cv2.imread(src_path, 0)
# tag_img = cv2.imread(tag_path, 0)
#
# src_minx, src_miny, src_minw, src_minh = addBoundingBox(src_img)
# tag_minx, tag_miny, tag_minw, tag_minh = addBoundingBox(tag_img)
#
# src_min_bounding = src_img[src_miny: src_miny+src_minh, src_minx: src_minx+src_minw]
# tag_min_bounding = tag_img[tag_miny: tag_miny+tag_minh, tag_minx: tag_minx+tag_minw]
#
# src_maxw = max(src_minw, src_minh)
# tag_maxw = max(tag_minw, tag_minh)
#
# src_new_square = np.ones((src_maxw, src_maxw)) * 255
# tag_new_square = np.ones((tag_maxw, tag_maxw)) * 255
#
# # new src square
# for y in range(src_min_bounding.shape[0]):
#     for x in range(src_min_bounding.shape[1]):
#         if src_min_bounding.shape[0] > src_min_bounding.shape[1]:
#             # height > width
#             offset = int((src_min_bounding.shape[0] - src_min_bounding.shape[1]) / 2)
#             src_new_square[y][x+offset] = src_min_bounding[y][x]
#         else:
#             # height < width
#             offset = int((src_min_bounding.shape[1] - src_min_bounding.shape[0]) / 2)
#             src_new_square[y+offset][x] = src_min_bounding[y][x]
#
#
# # new tag square
# for y in range(tag_min_bounding.shape[0]):
#     for x in range(tag_min_bounding.shape[1]):
#         if tag_min_bounding.shape[0] > tag_min_bounding.shape[1]:
#             # height > width
#             offset = int((tag_min_bounding.shape[0] - tag_min_bounding.shape[1]) / 2)
#             tag_new_square[y][x+offset] = tag_min_bounding[y][x]
#         else:
#             # height < width
#             offset = int((tag_min_bounding.shape[1] - tag_min_bounding.shape[0]) / 2)
#             tag_new_square[y+offset][x] = tag_min_bounding[y][x]
#
# # resize new square to same size between the source image and target image
# if src_new_square.shape[0] > tag_new_square.shape[0]:
#     # src > tag
#     src_new_square = cv2.resize(src_new_square, tag_new_square.shape)
# else:
#     # src < tag
#     tag_new_square = cv2.resize(tag_new_square, src_new_square.shape)
#
# # histogram
# # plt.hist(src_new_square.ravel(), 256, [0, 256]); plt.show()
# # plt.hist(tag_new_square.ravel(), 256, [0, 256]); plt.show()
#
# # x-axis and y-axis statistics histogram
# src_x_hist = np.zeros(src_new_square.shape[1])
# src_y_hist = np.zeros(src_new_square.shape[0])
#
# tag_x_hist = np.zeros(tag_new_square.shape[1])
# tag_y_hist = np.zeros(tag_new_square.shape[0])
#
# for y in range(src_new_square.shape[0]):
#     for x in range(src_new_square.shape[1]):
#         if src_new_square[y][x] == 0:
#             src_y_hist[y] += 1
#             src_x_hist[x] += 1
#
# for y in range(tag_new_square.shape[0]):
#     for x in range(tag_new_square.shape[1]):
#         if tag_new_square[y][x] == 0:
#             tag_y_hist[y] += 1
#             tag_x_hist[x] += 1
#
# print(src_x_hist)
# print(src_y_hist)
#
# plt.subplot(221); plt.plot(src_x_hist)
# plt.subplot(222); plt.plot(src_y_hist)
# plt.subplot(223); plt.plot(tag_x_hist)
# plt.subplot(224); plt.plot(tag_y_hist)
#
# plt.show()
#
#
# # img_file = "../characters/tag_bing copy.png.png"
# #
# # img = cv2.imread(img_file, 0)
# #
# # print(img.shape)
# #
# # rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# #
# # WIDTH = img.shape[0]
# # HEIGHT = img.shape[1]
# #
# #
# #
# # # moments
# # im2, contours, hierarchy = cv2.findContours(img, 1, 2)
# # print("Contours len: %s " % len(contours))
# #
# # cnt = contours[0]
# # M = cv2.moments(cnt)
# # # print(M)
# #
# # # center of mass
# # cx = int(M['m10'] / M['m00'])
# # cy = int(M['m01'] / M['m00'])
# # print('( %d, %d)' % (cy, cx))
# #
# #
# # # area
# # area = cv2.contourArea(cnt)
# #
# # print("Area:", area)
# #
# # minx = WIDTH
# # miny = HEIGHT
# # maxx = 0
# # maxy = 0
# # # Bounding box
# # for i in range(len(contours)):
# #     x, y, w, h = cv2.boundingRect(contours[i])
# #     if w > 0.95 * WIDTH and h > 0.95 * HEIGHT:
# #         continue
# #
# #     if x < minx:
# #         minx = x
# #     if y < miny:
# #         miny = y
# #     if x+w > maxx:
# #         maxx = x+w
# #     if y+h > maxy:
# #         maxy = y+h
# #
# #     cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (0,255,0), 2)
# #
# # cv2.rectangle(rgb_img, (minx, miny), (maxx, maxy), (255, 0, 0), 3)
# #
# # min_bound_width = maxx - minx + 1;
# # min_bound_height = maxy - miny + 1;
# #
# # aspect_ratio = min_bound_height / min_bound_width * 1.0
# # print("Aspect ratio: %f \n" % aspect_ratio)
#
#
# # convex hull
#
#
# # cv2.imshow("src", src_new_square)
# # cv2.imshow("tag", tag_new_square)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# a = [[0, 3, 4, 8], [1, 2, 5, 6], [2, 1, 6], [3, 0, 4, 8], [4, 0, 3, 8, 10], [5, 1, 6], [6, 1, 2, 5], [7, 9], [8, 0, 3, 4, 10], [9, 7], [10, 4, 8], [11], [12, 13, 14, 20], [13, 12, 14, 20], [14, 12, 13, 20, 28], [15, 21, 22, 26], [16, 18, 23, 30], [17, 19, 24, 25], [18, 16, 23, 27, 30], [19, 17, 24, 25], [20, 12, 13, 14, 28, 29], [21, 15, 22, 26], [22, 15, 21, 26], [23, 16, 18, 27, 30], [24, 17, 19, 25], [25, 17, 19, 24], [26, 15, 21, 22], [27, 18, 23, 30], [28, 14, 20, 29], [29, 20, 28], [30, 16, 18, 23, 27, 33], [31, 32, 34, 37, 39, 40], [32, 31, 37, 39, 40], [33, 30, 35, 38, 42], [34, 31, 37, 39, 40, 45], [35, 33, 38, 42], [36, 41, 43], [37, 31, 32, 34, 39, 40, 45], [38, 33, 35, 42, 46], [39, 31, 32, 34, 37, 40, 45], [40, 31, 32, 34, 37, 39, 45], [41, 36, 43], [42, 33, 35, 38, 46], [43, 36, 41], [44], [45, 34, 37, 39, 40], [46, 38, 42], [47, 52, 55], [48, 50], [49, 51], [50, 48, 56], [51, 49], [52, 47, 55], [53, 54], [54, 53], [55, 47, 52], [56, 50], [57, 58, 59, 62], [58, 57, 59, 62], [59, 57, 58, 62], [60, 61, 64], [61, 60, 64, 67], [62, 57, 58, 59, 66], [63, 65], [64, 60, 61, 67], [65, 63], [66, 62], [67, 61, 64], [68], [69, 70, 72, 75], [70, 69, 72, 75, 78], [71, 76], [72, 69, 70, 75, 78], [73, 74, 77], [74, 73, 77], [75, 69, 70, 72, 78], [76, 71], [77, 73, 74], [78, 70, 72, 75], [79], [80]]
#
# final_clustor = []
# used_index = []
# for i in range(len(a)):
#     if i in used_index:
#         continue
#
#     new_clustor = a[i]
#
#     for j in range(i+1, len(a)):
#         if len(set(new_clustor).intersection(set(a[j]))) == 0:
#             continue
#         new_clustor = list(set(new_clustor).union(set(a[j])))
#         used_index.append(j)
#     final_clustor.append(new_clustor)
#
# print(final_clustor)
#
# a = [0, 3, 4, 8]
# b = [1,2,5,6]
# c = [8,0,3,4,10]
#
# print(list(set(a).intersection(set(b))))
# print(set(a).intersection(set(c)))


import cv2
import numpy as np
from utils.Functions import getBoundingBoxes
import math


def main():
    pass
