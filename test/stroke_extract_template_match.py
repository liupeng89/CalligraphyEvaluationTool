# coding: utf-8
import cv2
import numpy as np
import math

from utils.Functions import getContourImage, sortPointsOnContourOfImage, getConnectedComponents, createBlankRGBImage, \
    getSkeletonOfImage, getCrossPointsOfSkeletonLine, getEndPointsOfSkeletonLine,removeBranchOfSkeletonLine
from utils.stroke_extraction_algorithm import autoStrokeExtractFromCharacter, removeShortBranchesOfSkeleton
from algorithms.RDP import rdp
from utils.contours_smoothed_algorithm import autoSmoothContoursOfCharacter


path = "../test_images/page1_char_3.png"
img = cv2.imread(path, 0)

_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# get contours
img_contour = getContourImage(img)

# get skeleton
img_skel = getSkeletonOfImage(img)

# corner points
contours = getConnectedComponents(img_contour, connectivity=8)
print("contours num:", len(contours))

corner_points = []

for cont in contours:
    cont_sorted = sortPointsOnContourOfImage(cont)

    cont_points = rdp(cont_sorted, 5)
    corner_points.append(cont_points)

print("corner points num:", len(corner_points))

img_contour_rgb = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2RGB)

# for i in range(len(corner_points)):
#     for pt in corner_points[i]:
#         cv2.circle(img_contour_rgb, pt, 3, (0,0,255), 4 )

# merged corner points
CORNER_CROSS_DIST_THRESHOLD = 20
corner_points_merged = []
#
for k in range(len(corner_points)):
    corners_points_merged = []
    for i in range(len(corner_points[k])):
        pt = corner_points[k][i]
        is_valid = True
        closed_points = []
        for j in range(i+1, len(corner_points[k])):
            cpt = corner_points[k][j]
            dist = math.sqrt((pt[0] - cpt[0]) ** 2 + (pt[1] - cpt[1]) ** 2)
            if dist < CORNER_CROSS_DIST_THRESHOLD:
                is_valid = False
                closed_points.append(cpt)
                break
        if is_valid:
            corners_points_merged.append(pt)
        else:
            corners_points_merged.append(closed_points[-1])
    corner_points_merged.append(corners_points_merged)


#
#
# # for i in range(len(corner_points)):
# #     pt = corner_points[i]
# #     for j in range(len(corner_points)):
# #         if i == j:
# #             continue
# #         cpt = corner_points[j]
# #         dist = math.sqrt((pt[0] - cpt[0]) ** 2 + (pt[1] - cpt[1]) ** 2)
# #         if dist < CORNER_CROSS_DIST_THRESHOLD:
# #             corners_points_merged.append(pt)
# #             break
#
# # for pt in corner_points:
# #     for cpt in corner_points:
# #         dist = math.sqrt((pt[0] - cpt[0]) ** 2 + (pt[1] - cpt[1]) ** 2)
# #         if dist < CORNER_CROSS_DIST_THRESHOLD:
# #             corners_points_merged.append(pt)
# #             break

for i in range(len(contours)):
    cont = contours[i]
    cont_sorted = sortPointsOnContourOfImage(cont)

    corners = corner_points_merged[i]
    print(corners)

    sub_contours = []
    for j in range(len(corners)):
        start_pt = corners[j]
        if j == len(corners) - 1:
            end_pt = corners[0]
        else:
            end_pt = corners[j+1]

        start_index = cont_sorted.index(start_pt)
        end_index = cont_sorted.index(end_pt)

        sub_cont = cont_sorted[start_index: end_index+1]
        sub_contours.append(sub_cont)

    print(len(sub_contours))

    for k in range(len(sub_contours)):
        color = (255, 0, 0)
        if k % 3 == 1:
            color = (0, 255, 0)
        elif k % 3 == 2:
            color = (0,0, 255)
        for pt in sub_contours[k]:
            img_contour_rgb[pt[1]][pt[0]] = color


for i in range(len(corner_points)):
    for pt in corner_points[i]:
        if pt in corner_points_merged[i]:
            cv2.circle(img_contour_rgb, pt, 3, (255,0,0), 4 )
        else:
            cv2.circle(img_contour_rgb, pt, 3, (0, 0, 255), 4)

cv2.imshow("img", img)
cv2.imshow("img contour", img_contour_rgb)
cv2.imshow("img skel", img_skel)

cv2.waitKey(0)
cv2.destroyAllWindows()