import numpy as np
import cv2
from matplotlib import pyplot as plt

from functions.AddBoundingBox import addBoundingBox

src_path = "../characters/src_dan_processed.png"
tag_path = "../characters/tag_dan_processed.png"

src_img = cv2.imread(src_path, 0)
tag_img = cv2.imread(tag_path, 0)

src_minx, src_miny, src_minw, src_minh = addBoundingBox(src_img)
tag_minx, tag_miny, tag_minw, tag_minh = addBoundingBox(tag_img)

src_min_bounding = src_img[src_miny: src_miny+src_minh, src_minx: src_minx+src_minw]
tag_min_bounding = tag_img[tag_miny: tag_miny+tag_minh, tag_minx: tag_minx+tag_minw]

src_maxw = max(src_minw, src_minh)
tag_maxw = max(tag_minw, tag_minh)

src_new_square = np.ones((src_maxw, src_maxw)) * 255
tag_new_square = np.ones((tag_maxw, tag_maxw)) * 255

# new src square
for y in range(src_min_bounding.shape[0]):
    for x in range(src_min_bounding.shape[1]):
        if src_min_bounding.shape[0] > src_min_bounding.shape[1]:
            # height > width
            offset = int((src_min_bounding.shape[0] - src_min_bounding.shape[1]) / 2)
            src_new_square[y][x+offset] = src_min_bounding[y][x]
        else:
            # height < width
            offset = int((src_min_bounding.shape[1] - src_min_bounding.shape[0]) / 2)
            src_new_square[y+offset][x] = src_min_bounding[y][x]


# new tag square
for y in range(tag_min_bounding.shape[0]):
    for x in range(tag_min_bounding.shape[1]):
        if tag_min_bounding.shape[0] > tag_min_bounding.shape[1]:
            # height > width
            offset = int((tag_min_bounding.shape[0] - tag_min_bounding.shape[1]) / 2)
            tag_new_square[y][x+offset] = tag_min_bounding[y][x]
        else:
            # height < width
            offset = int((tag_min_bounding.shape[1] - tag_min_bounding.shape[0]) / 2)
            tag_new_square[y+offset][x] = tag_min_bounding[y][x]

# resize new square to same size between the source image and target image
if src_new_square.shape[0] > tag_new_square.shape[0]:
    # src > tag
    src_new_square = cv2.resize(src_new_square, tag_new_square.shape)
else:
    # src < tag
    tag_new_square = cv2.resize(tag_new_square, src_new_square.shape)

# histogram
# plt.hist(src_new_square.ravel(), 256, [0, 256]); plt.show()
# plt.hist(tag_new_square.ravel(), 256, [0, 256]); plt.show()

# x-axis and y-axis statistics histogram
src_x_hist = np.zeros(src_new_square.shape[1])
src_y_hist = np.zeros(src_new_square.shape[0])

tag_x_hist = np.zeros(tag_new_square.shape[1])
tag_y_hist = np.zeros(tag_new_square.shape[0])

for y in range(src_new_square.shape[0]):
    for x in range(src_new_square.shape[1]):
        if src_new_square[y][x] == 0:
            src_y_hist[y] += 1
            src_x_hist[x] += 1

for y in range(tag_new_square.shape[0]):
    for x in range(tag_new_square.shape[1]):
        if tag_new_square[y][x] == 0:
            tag_y_hist[y] += 1
            tag_x_hist[x] += 1

print(src_x_hist)
print(src_y_hist)

plt.subplot(221); plt.plot(src_x_hist)
plt.subplot(222); plt.plot(src_y_hist)
plt.subplot(223); plt.plot(tag_x_hist)
plt.subplot(224); plt.plot(tag_y_hist)

plt.show()


# img_file = "../characters/tag_bing copy.png.png"
#
# img = cv2.imread(img_file, 0)
#
# print(img.shape)
#
# rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
# WIDTH = img.shape[0]
# HEIGHT = img.shape[1]
#
#
#
# # moments
# im2, contours, hierarchy = cv2.findContours(img, 1, 2)
# print("Contours len: %s " % len(contours))
#
# cnt = contours[0]
# M = cv2.moments(cnt)
# # print(M)
#
# # center of mass
# cx = int(M['m10'] / M['m00'])
# cy = int(M['m01'] / M['m00'])
# print('( %d, %d)' % (cy, cx))
#
#
# # area
# area = cv2.contourArea(cnt)
#
# print("Area:", area)
#
# minx = WIDTH
# miny = HEIGHT
# maxx = 0
# maxy = 0
# # Bounding box
# for i in range(len(contours)):
#     x, y, w, h = cv2.boundingRect(contours[i])
#     if w > 0.95 * WIDTH and h > 0.95 * HEIGHT:
#         continue
#
#     if x < minx:
#         minx = x
#     if y < miny:
#         miny = y
#     if x+w > maxx:
#         maxx = x+w
#     if y+h > maxy:
#         maxy = y+h
#
#     cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (0,255,0), 2)
#
# cv2.rectangle(rgb_img, (minx, miny), (maxx, maxy), (255, 0, 0), 3)
#
# min_bound_width = maxx - minx + 1;
# min_bound_height = maxy - miny + 1;
#
# aspect_ratio = min_bound_height / min_bound_width * 1.0
# print("Aspect ratio: %f \n" % aspect_ratio)


# convex hull


# cv2.imshow("src", src_new_square)
# cv2.imshow("tag", tag_new_square)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()