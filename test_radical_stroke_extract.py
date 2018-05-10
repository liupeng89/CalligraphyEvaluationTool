import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeBranchOfSkeleton, getEndPointsOfSkeletonLine, \
                          getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints, createBlankGrayscaleImage, \
                            getLinePoints, getBreakPointsFromContour, merge_corner_lines_to_point, getCropLines, \
                            getCornerPointsOfImage, getClusterOfCornerPoints, getCropLinesPoints

# 1133壬 2252支 0631叟 0633口
path = "1133壬.jpg"
img = cv2.imread(path)

# contour
contour = getContourOfImage(img)
contour = getSkeletonOfImage(contour)
contour = np.array(contour, dtype=np.uint8)

img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

components = getConnectedComponents(img)
print("radicals num: %d" % len(components))

radicals = components[0]
radicals = np.array(radicals, dtype=np.uint8)

# sorted the contour points
contour_sorted = sortPointsOnContourOfImage(contour)
contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)

# # skeleton
skeleton = getSkeletonOfImage(radicals)
# # remove extra branches
skeleton = removeBranchOfSkeleton(skeleton, distance_threshod=20)
#
end_points = getEndPointsOfSkeletonLine(skeleton)
cross_points = getCrossPointsOfSkeletonLine(skeleton)
skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)

# corner area detect
corner_points = getCornerPointsOfImage(img.copy(), contour, cross_points, end_points)

for pt in corner_points:
    contour_rgb[pt[1]][pt[0]] = (0, 0, 255)
print("corner points num: %d" % len(corner_points))

# cluster the corner points
dist_threshold = 30
corner_points_cluster = getClusterOfCornerPoints(corner_points, cross_points)
print("corner cluster num:%d" % len(corner_points_cluster))

# detect corner points type: two point, four point (rectangle or diamond)
crop_lines = getCropLines(corner_points_cluster)
for line in crop_lines:
    cv2.line(contour_rgb, line[0], line[1], (0, 255, 0), 1)

# crop lines points
crop_lines_points = getCropLinesPoints(img, crop_lines)
print("crop lines num: %d" % len(crop_lines_points))

img_separate = img.copy()
for line in crop_lines:
    cv2.line(img_separate, line[0], line[1], 255, 1)
# for line in crop_lines_points:
#     if line is None:
#         continue
#     for pt in line:
#         img_separate[pt[1]][pt[0]] = 255

strokes_components = getConnectedComponents(img_separate, connectivity=8)
print("storkes components num: %d" % len(strokes_components))

# find region based on the crop lines
sub_contours = segmentContourBasedOnCornerPoints(contour_sorted, corner_points)
print("sub contours num: %d" % len(sub_contours))



for i in range(len(strokes_components)):
    component = strokes_components[i]
    cv2.imshow("component_%d" % i, component)
# recompose strokes components

contour_separate_region = cv2.cvtColor(contour_rgb, cv2.COLOR_RGB2GRAY)
_, contour_separate_region = cv2.threshold(contour_separate_region, 240, 255, cv2.THRESH_BINARY)


# seprarate contour
# sub_contours_cluster = []
# used_index = []
# for i in range(len(sub_contours)):
#     if i in used_index:
#         continue
#     used_index.append(i)
#     c_start_pt = sub_contours[i][0]
#     c_end_pt = sub_contours[i][-1]
#
#     sub_contour = [sub_contours[i]]
#
#     for line in crop_lines:
#         index = 0
#         if c_start_pt in line and c_end_pt not in line:
#             index = line.index(c_start_pt)
#         elif c_start_pt not in line and c_end_pt in line:
#             index = line.index(c_end_pt)
#         print(index)
#
#         if index == 0:
#             next_pt = line[1]
#         elif index == 1:
#             next_pt = line[0]
#         for j in range(len(sub_contours)):
#             if i == j or j in used_index:
#                 continue
#             if next_pt in sub_contours[j] and line[index] not in sub_contours[j]:
#                 sub_contour.append(sub_contours[j])
#                 used_index.append(j)
#
#     sub_contours_cluster.append(sub_contour)
#
# print(sub_contours_cluster)

# for i in range(len(sub_contours_cluster)):
#     bk = createBlankGrayscaleImage(img)
#
#     for sub in sub_contours_cluster[i]:
#         for pt in sub:
#             bk[pt[1]][pt[0]] = 0
#     cv2.imshow("bk%d" % i, bk)

# for i in range(len(sub_contours)):
#     bk = createBlankGrayscaleImage(img)
#
#     for pt in sub_contours[i]:
#         bk[pt[1]][pt[0]] = 0
#
#     cv2.imshow("bk_%d" % i, bk)
#
# for i in range(len(crop_lines)):
#     bk = createBlankGrayscaleImage(img)
#
#     cv2.line(bk, crop_lines[i][0], crop_lines[i][1], 0, 1)
#
#     cv2.imshow("line_%d"%i, bk)

sub_contours_cluster = []
sub_contours_cluster.append([0, 2,4,6])
sub_contours_cluster.append([1,5])
sub_contours_cluster.append([3])
sub_contours_cluster.append([7])

crop_lines_cluster = []
crop_lines_cluster.append([0,1, 2, 5])
crop_lines_cluster.append([3, 4])
crop_lines_cluster.append([5])
crop_lines_cluster.append([0])

for i in range(len(sub_contours_cluster)):
    sub_cluster = sub_contours_cluster[i]
    line_cluster = crop_lines_cluster[i]

    bk = createBlankGrayscaleImage(img)

    for index in sub_cluster:
        sub = sub_contours[index]
        for pt in sub:
            bk[pt[1]][pt[0]] = 0

    for index in line_cluster:
        line = crop_lines[index]
        cv2.line(bk, line[0], line[1], 0, 1)

    # fill color
    cont_sorted = sortPointsOnContourOfImage(bk)
    cont_sorted = np.array([cont_sorted], "int32")
    stroke_img = createBlankGrayscaleImage(bk)
    stroke_img = cv2.fillPoly(stroke_img, cont_sorted, 0)


    cv2.imshow("stroke_%d" % i, stroke_img)





# cv2.imshow("radicals", radicals)
cv2.imshow("contour", contour)
cv2.imshow("skeleton", skeleton)
# cv2.imshow("skeleton rgb", skeleton_rgb)
# cv2.imshow("img_corner_area", img_corner_area)
# cv2.imshow("contour_rgb", contour_rgb)
cv2.imshow("contour_separate_region", contour_separate_region)
# cv2.imshow("contour_separate_region_bit",  contour_separate_region_bit)
cv2.imshow("img_separate", img_separate)
cv2.waitKey(0)
cv2.destroyAllWindows()



# exit()
#
# def isInCluster(point_pair, cluster):
#     if point_pair is None or cluster is None:
#         return False
#     label = False
#     for cl in cluster:
#         if point_pair[0] in cl and point_pair[1] in cl:
#             label = True
#             break
#     return label
#
#
# # segment contour to sub-contour
# print("contor point num: %d" % len(contour_sorted))
# sub_contours = segmentContourBasedOnCornerPoints(contour_sorted, corner_points)
# print("sub contour num: %d" % len(sub_contours))
#
# # corner points correspondence
#
#
# def isInOneSubContour(pt1, pt2, sub_contours):
#     if pt1 is None or pt2 is None or sub_contours is None:
#         return False
#     label = False
#     for sub in sub_contours:
#         if pt1 in sub and pt2 in sub:
#             label = True
#             break
#     return label
#
# # co-linear  |y1-y2| <= 10 pixels and not in same sub-contour
# co_linear_points = []
# parallel_points = []
# co_sub_contour = []
# for i in range(len(corner_points)):
#     pt1 = corner_points[i]
#     for j in range(len(corner_points)):
#         if i == j:
#             continue
#         pt2 = corner_points[j]
#
#         # co-linear should be in same cluster and can be in same sub-contour
#         if abs(pt1[0] - pt2[0]) <= 10 and isInCluster([pt1, pt2], corner_points_cluster):
#             # co-linear
#             if [pt1, pt2] not in co_linear_points and [pt2, pt1] not in co_linear_points:
#                 co_linear_points.append([pt1, pt2])
#
#         # parallel, should not be in same sub-contour, but should be in same cluster
#         if abs(pt1[1] - pt2[1]) <= 10 and not isInOneSubContour(pt1, pt2, sub_contours) and isInCluster([pt1, pt2], corner_points_cluster):
#             # parallel
#             if [pt1, pt2] not in parallel_points and [pt2, pt1] not in parallel_points:
#                 if [pt1, pt2] not in co_linear_points and [pt2, pt1] not in co_linear_points:
#                     parallel_points.append([pt1, pt2])
#
#         # co sub-contour, and do not repeat in previous lists.
#         if isInOneSubContour(pt1, pt2, sub_contours):
#             # co subcontour
#             if [pt1, pt2] not in co_sub_contour and [pt2, pt1] not in co_sub_contour:
#                 if [pt1, pt2] not in co_linear_points and [pt2, pt1] not in co_linear_points:
#                     if [pt1, pt2] not in parallel_points and [pt2, pt1] not in parallel_points and isInCluster([pt1, pt2], corner_points_cluster):
#                         co_sub_contour.append([pt1, pt2])
#
# print(co_linear_points)
# print(parallel_points)
# print(co_sub_contour)
#
# co_linear_points_cluster = []
# parallel_points_cluster = []
# co_sub_contour_cluster = []
#
# def isSubList(l1, l2):
#     if l1 == [] and l2 != []:
#         return True
#     if l1 != [] and l2 == []:
#         return False
#     if l1 == [] and l2 == []:
#         return True
#     for item in l1:
#         if item not in l2:
#             return False
#     return True
#
# # cluster co_linear points pair based on the cluster points
# used_index = []
# for i in range(len(co_linear_points)):
#     if i in used_index:
#         continue
#     used_index.append(i)
#     pair_cluster = [co_linear_points[i]]
#     cluster = None
#     for cl in corner_points_cluster:
#         if isSubList(co_linear_points[i], cl):
#             cluster = cl.copy()
#
#     # j
#     if cluster is None:
#         print("cluster should not be None!")
#     for j in range(len(co_linear_points)):
#         if j == i or j in used_index:
#             continue
#         if isSubList(co_linear_points[j], cluster):
#             pair_cluster.append(co_linear_points[j])
#         used_index.append(j)
#     co_linear_points_cluster.append(pair_cluster)
#
# print(co_linear_points_cluster)
#
# # cluster the parallel points pair
# used_index = []
# for i in range(len(parallel_points)):
#     if i in used_index:
#         continue
#     used_index.append(i)
#     pair_cluster = [parallel_points[i]]
#     cluster = None
#     for cl in corner_points_cluster:
#         if isSubList(parallel_points[i], cl):
#             cluster = cl.copy()
#
#     # j
#     if cluster is None:
#         print("cluster should not be None!")
#     for j in range(len(parallel_points)):
#         if j == i or j in used_index:
#             continue
#         if isSubList(parallel_points[j], cluster):
#             pair_cluster.append(parallel_points[j])
#         used_index.append(j)
#     parallel_points_cluster.append(pair_cluster)
#
# print(parallel_points_cluster)
#
# # cluster the co-subcontour points pair
# used_index = []
# for i in range(len(co_sub_contour)):
#     if i in used_index:
#         continue
#     used_index.append(i)
#     pair_cluster = [co_sub_contour[i]]
#     cluster = None
#     for cl in corner_points_cluster:
#         if isSubList(co_sub_contour[i], cl):
#             cluster = cl.copy()
#
#     # j
#     if cluster is None:
#         print("cluster should not be None!")
#     for j in range(len(co_sub_contour)):
#         if j == i or j in used_index:
#             continue
#         if isSubList(co_sub_contour[j], cluster):
#             pair_cluster.append(co_sub_contour[j])
#         used_index.append(j)
#         co_sub_contour_cluster.append(pair_cluster)
#
# print(co_sub_contour_cluster)
#
# def findCoLinearSubContours(point_pair, sub_contours):
#     """
#     Find two co-linear sub-contours based on the point pair.
#     :param point_pair:
#     :param sub_contours:
#     :return:
#     """
#     if point_pair is None or sub_contours is None:
#         return
#     pt1 = point_pair[0]; pt2 = point_pair[1]
#     sub1 = None; sub2 = None
#     for sub in sub_contours:
#         if pt1 in sub and pt2 not in sub:
#             sub1 = sub.copy()
#         if pt2 in sub and pt1 not in sub:
#             sub2 = sub.copy()
#     return [sub1, sub2]
#
# #
# print("sub-contours num : %d" % len(sub_contours))
#
# # def findCoSubContour(pair, sub_contours):
# #     if pair is None or sub_contours is None:
# #         return
# #     for sub in sub_contours:
# #         if pair[0] in sub
# #
# # for pair in co_sub_contour:
# #     pass
#
#



#
#
#
# # crop character
# print("contor point num: %d" % len(contour_sorted))
# sub_contours = segmentContourBasedOnCornerPoints(contour_sorted, corner_points)
# print("sub contours num: %d" % len(sub_contours))
#
# # separate single region to several region
# contour_separate_region = cv2.cvtColor(contour_rgb, cv2.COLOR_RGB2GRAY)
# _, contour_separate_region = cv2.threshold(contour_separate_region, 240, 255, cv2.THRESH_BINARY)
#
# contour_separate_region_bit = np.array(255-contour_separate_region, dtype=np.uint8)

# for i in range(len(sub_contours)):
#     sub = sub_contours[i]
#     bk_img = createBlankGrayscaleImage(img)
#     for pt in sub:
#         bk_img[pt[1]][pt[0]] = 0
#
#     cv2.imshow("sub contour_%d" % i, bk_img)


# for y in range(radicals.shape[0]):
#     for x in range(radicals.shape[1]):
#         if radicals[y][x] == 255:
#             contour_separate_region_bit[y][x] = 255
#
# stroke_components = getConnectedComponents(contour_separate_region_bit)
# print(len(stroke_components))

