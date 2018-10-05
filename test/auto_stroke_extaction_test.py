# coding: utf-8
import cv2
import numpy as np
from algorithms.RDP import rdp
import math

from utils.Functions import getConnectedComponents, getSkeletonOfImage, getEndPointsOfSkeletonLine, \
                            getCrossPointsOfSkeletonLine, createBlankGrayscaleImage, getCropLines, \
                            getClusterOfCornerPoints, getAllMiniBoundingBoxesOfImage, getContourImage, \
                            getValidCornersPoints, getDistanceBetweenPointAndComponent, isValidComponent, \
                            removeShortBranchesOfSkeleton, sortPointsOnContourOfImage, removeBreakPointsOfContour

from utils.stroke_extraction_algorithm import autoStrokeExtractFromCharacter


path = "../test_images/page1_char_3.png"

# Image porcessing
img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

components = getConnectedComponents(img, connectivity=8)
print("components num: %d" % len(components))

# 6. Get skeletons of component.
comp_skeleton = getSkeletonOfImage(components[2])
comp_skeleton = getSkeletonOfImage(comp_skeleton)
cv2.imshow("skeleton_original", comp_skeleton)

# 7. Process the skeleton by remove extra branches.
# comp_skeleton = removeShortBranchesOfSkeleton(comp_skeleton, length_threshold=30)
comp_skeleton_rgb = cv2.cvtColor(comp_skeleton, cv2.COLOR_GRAY2RGB)
cv2.imshow("skeleton_smoothed", comp_skeleton)

# 8. Get the end points and cross points after skeleton processed
end_points = getEndPointsOfSkeletonLine(comp_skeleton)
cross_points = getCrossPointsOfSkeletonLine(comp_skeleton)
print("end points num: %d ,and cross points num: %d" % (len(end_points), len(cross_points)))

# 9. Get contour image of component
comp_contours_img = getContourImage(components[2])
comp_contours_img_rbg = cv2.cvtColor(comp_contours_img, cv2.COLOR_GRAY2RGB)

# 10. Detect the number of contours and return all contours
comp_contours = getConnectedComponents(comp_contours_img, connectivity=8)
print("contours num: %d" % len(comp_contours))


# 11. Get points on contours
corners_points = []
for cont in comp_contours:
    cont = removeBreakPointsOfContour(cont)
    cont_sorted = sortPointsOnContourOfImage(cont)
    cont_points = rdp(cont_sorted, 5)
    corners_points += cont_points

CORNER_CROSS_DIST_THRESHOLD = 20
corners_points_merged = []
for pt in corners_points:
    for cpt in cross_points:
        dist = math.sqrt((pt[0] - cpt[0])**2+(pt[1] - cpt[1])**2)
        if dist < CORNER_CROSS_DIST_THRESHOLD:
            corners_points_merged.append(pt)
            break

# for pt in cross_points_:
#     cv2.circle(comp_skeleton_rgb, pt, 2, (0,255,0), 3)
for pt in cross_points:
    cv2.circle(comp_skeleton_rgb, pt, 2, (0, 0, 255), 3)
for pt in end_points:
    cv2.circle(comp_skeleton_rgb, pt, 2, (255, 0, 0), 3)
cv2.imshow("comp_skeleton_rgb", comp_skeleton_rgb)

for pt in corners_points:
    cv2.circle(comp_contours_img_rbg, pt, 4, (0,0,255), 4)
for pt in corners_points_merged:
    cv2.circle(comp_contours_img_rbg, pt, 2, (0, 255, 0), 3)
cv2.imshow("comp_contours_img_rbg", comp_contours_img_rbg)

corners_points_cluster = getClusterOfCornerPoints(corners_points_merged, corners_points,50)
print(corners_points_cluster)


# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# auto extract strokes
# strokes = autoStrokeExtractFromCharacter(img)
#
# print("stroke num:", len(strokes))
#
# for i in range(len(strokes)):
#     cv2.imshow("stroke_%d" % i, strokes[i])


# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # img_rbg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# img_rgb = createBlankRGBImage(img)
#
#
# # Get the contour of image
# img_contour = getContourImage(img)
#
# # Sorted the points on contour
# img_contour_points = sortPointsOnContourOfImage(img_contour)
# print("Before simplified num:", len(img_contour_points))
#
# # Simplify points on contour
# img_contour_points = rdp(img_contour_points, 3)
# print("After simplified num: ", len(img_contour_points))
#
# # Start point
# cv2.circle(img_rgb, img_contour_points[0], 5, (0, 255, 0), 6)
#
# for i in range(len(img_contour_points)):
#     start_pt = img_contour_points[i]
#     if i == len(img_contour_points) - 1:
#         end_pt = img_contour_points[0]
#     else:
#         end_pt = img_contour_points[i + 1]
#     cv2.line(img_rgb, start_pt, end_pt, (0, 0, 255), 1)
#
#     cv2.circle(img_rgb, start_pt, 4, (255, 0, 0), 5)
#
# # Merge closed points
# img_contour_points_merged = [img_contour_points[0]]
# POINTS_DIST_THRESHOLD = 11
# for i in range(len(img_contour_points)):
#     curr_pt = img_contour_points[i]
#     if i == len(img_contour_points) - 1:
#         next_pt = img_contour_points[0]
#     else:
#         next_pt = img_contour_points[i + 1]
#     dist = math.sqrt((curr_pt[0] - next_pt[0])**2 + (curr_pt[1] - next_pt[1])**2)
#
#     if dist < POINTS_DIST_THRESHOLD:
#         img_contour_points_merged.append(next_pt)
#         i += 1
#     else:
#         img_contour_points_merged.append(curr_pt)
# print("After merged num:", len(img_contour_points_merged))
# for pt in img_contour_points_merged:
#     cv2.circle(img_rgb, pt, 2, (0, 0, 255), 2)




# Get each point tangent angle
# for i in range(len(img_contour_points) - 1):
#     start_pt = img_contour_points[i]
#     mid_pt = end_pt = start_pt
#     if i == len(img_contour_points) - 2:
#         mid_pt = img_contour_points[i + 1]
#         end_pt = img_contour_points[0]
#     elif i == len(img_contour_points) - 1:
#         mid_pt = img_contour_points[0]
#         end_pt = img_contour_points[1]
#
#     # calculate the angle between Line(start_pt, mid_pt) and Line(mid_pt, end_pt)
#     theta = math.atan2(end_pt[1] - mid_pt[1], end_pt[0] - mid_pt[0]) - \
#             math.atan2(start_pt[1] - mid_pt[1], start_pt[0] - mid_pt[0])
#     print(theta)
#
#     if theta > 0.05 or theta < -0.05:
#         cv2.circle(img_rgb, start_pt, 3, (0, 255, 0), 4)






# cv2.imshow("img", img)
# cv2.imshow("contour", img_contour)
# cv2.imshow("rgb", img_rgb)
#
cv2.waitKey(0)
cv2.destroyAllWindows()
