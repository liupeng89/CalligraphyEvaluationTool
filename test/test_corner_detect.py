# coding: utf-8
import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeExtraBranchesOfSkeleton, getEndPointsOfSkeletonLine, \
                          getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints, createBlankGrayscaleImage, \
                            getLinePoints, getBreakPointsFromContour, merge_corner_lines_to_point, getCropLines, \
                            getCornerPointsOfImage, getClusterOfCornerPoints, getCropLinesPoints, \
                            getConnectedComponentsOfGrayScale, getAllMiniBoundingBoxesOfImage


path = "radical_gray.jpg"

img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# img_bit = cv2.bitwise_not(img)


ret, labels = cv2.connectedComponents(img, connectivity=4)
components = []
for r in range(1, ret):
    img_ = np.ones(img.shape, dtype=np.uint8) * 255
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if labels[y][x] == r:
                img_[y][x] = 0.0
    components.append(img_)

print("components num : %d" % len(components))


cv2.imshow("img", img)
cv2.imshow("img_bit", img)

for i in range(len(components)):
    cv2.imshow("component_%d" % i, components[i])


cv2.waitKey(0)
cv2.destroyAllWindows()




# contour_img = getContourOfImage(img_gray)
# contour_img = np.array(contour_img, dtype=np.uint8)

# rbg corner

# img_corner = np.float32(img_rgb.copy())
# dst = cv2.cornerHarris(img_corner, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
#
# corners_area_points = []
# for y in range(dst.shape[0]):
#     for x in range(dst.shape[1]):
#         if dst[y][x] > 0.1 * dst.max():
#             corners_area_points.append((x, y))
#
# for pt in corners_area_points:
#     img_rgb[y][x] = (0, 255, 0)

# # grayscale corner
# img_corner = np.float32(img_gray.copy())
# dst = cv2.cornerHarris(img_corner, 3, 3, 0.04)
# dst = cv2.dilate(dst, None)
#
# corners_area_points = []
# for y in range(dst.shape[0]):
#     for x in range(dst.shape[1]):
#         if dst[y][x] > 0.1 * dst.max():
#             corners_area_points.append((x, y))
# print(len(corners_area_points))
# # for pt in corners_area_points:
# #     img_rgb[pt[1]][pt[0]] = (255, 0, 0)
#
#
#
#
# # binary corner
# img_corner = np.float32(img_bit.copy())
# dst = cv2.cornerHarris(img_corner, 3, 3, 0.04)
# dst = cv2.dilate(dst, None)
#
# corners_area_points = []
# for y in range(dst.shape[0]):
#     for x in range(dst.shape[1]):
#         if dst[y][x] > 0.1 * dst.max():
#             corners_area_points.append((x, y))
# print(len(corners_area_points))
# for pt in corners_area_points:
#     img_rgb[pt[1]][pt[0]] = (0, 0, 255)
#
# corners_img = createBlankGrayscaleImage(img_gray)
# for pt in corners_area_points:
#     corners_img[pt[1]][pt[0]] = 0.0
#
# rectangles = getAllMiniBoundingBoxesOfImage(corners_img)
# print("rectangle num: %d" % len(rectangles))
#
# for rect in rectangles:
#     cv2.rectangle(img_rgb, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 1)
#
# corners_area_center_points = []
# for rect in rectangles:
#     corners_area_center_points.append((rect[0]+int(rect[2]/2.), rect[1]+int(rect[3]/2.)))
#     img_rgb[rect[1]+int(rect[3]/2.)][rect[0]+int(rect[2]/2.)] = (0, 255, 0)
# print("corner area num: %d" % len(corners_area_center_points))
#
# corners_points = []
# for pt in corners_area_center_points:
#     if contour_img[pt[1]][pt[0]] == 0.0:
#         corners_points.append(pt)
#     else:
#         min_dist = 100000
#         min_x = min_y = 0
#         for y in range(contour_img.shape[0]):
#             for x in range(contour_img.shape[1]):
#                 cpt = contour_img[y][x]
#                 if cpt == 0.0:
#                     dist = math.sqrt((x-pt[0])**2 + (y-pt[1])**2)
#                     if dist < min_dist:
#                         min_dist = dist
#                         min_x = x
#                         min_y = y
#         # points on contour
#         corners_points.append((min_x, min_y))
# print("corner points num: %d" % len(corners_points))
# for pt in corners_points:
#     img_rgb[pt[1]][pt[0]] = (255, 0, 0)
#
#
#
# cv2.imshow("rgb", img_rgb)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()