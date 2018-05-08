import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeBranchOfSkeleton, getEndPointsOfSkeletonLine, \
                          getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints

# 1133壬 2252支 0631叟
path = "2252支.jpg"

img = cv2.imread(path)

contour = getContourOfImage(img)

contour = getSkeletonOfImage(contour)


img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

components = getConnectedComponents(img)

print("radicals num: %d" % len(components))

radicals = components[0]
radicals = np.array(radicals, dtype=np.uint8)

# # contour
# contour = getContourOfImage(radicals, minVal=20, maxVal=240)
contour = np.array(contour, dtype=np.uint8)

contour_seg = getConnectedComponents(contour)
print("contourseg num %d" % len(contour_seg))

for i in range(len(contour_seg)):
    cv2.imshow("seg_%d" % i, contour_seg[i])


def getBreakPointsFromContour(contour):
    break_points = []
    if contour is None:
        return break_points
    # check whether exist break points or not
    start_pt = None
    for y in range(contour.shape[0]):
        for x in range(contour.shape[1]):
            if contour[y][x] == 0.0:
                if start_pt is None:
                    start_pt = (x, y)
                    break
        if start_pt is not None:
            break
    print("start point: (%d, %d)" % (start_pt[0], start_pt[1]))




for y in range(contour.shape[0]):
    for x in range(contour.shape[1]):
        if contour[y][x] == 0.0:
            num = getNumberOfValidPixels(contour, x, y)
            if num == 1:
                print("point: (%d, %d)" % (x, y))

#
#
# # # remove the break points
contour = removeBreakPointsOfContour(contour)
contour_sorted = sortPointsOnContourOfImage(contour)
contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
#
#
# # skeleton
skeleton = getSkeletonOfImage(radicals)
# # remove extra branches
skeleton = removeBranchOfSkeleton(skeleton, distance_threshod=20)
#
end_points = getEndPointsOfSkeletonLine(skeleton)
cross_points = getCrossPointsOfSkeletonLine(skeleton)
skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
#
for pt in end_points:
    skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)
for pt in cross_points:
    skeleton_rgb[pt[1]][pt[0]] = (0, 255, 0)

# corner area detect
img_corner = np.float32(img.copy())
dst = cv2.cornerHarris(img_corner, 3, 3, 0.03)
dst = cv2.dilate(dst, None)

corners_area_points = []
for y in range(dst.shape[0]):
    for x in range(dst.shape[1]):
        if dst[y][x] > 0.1 * dst.max():
            corners_area_points.append((x, y))
img_corner_area = img_rgb.copy()

for pt in corners_area_points:
    img_corner_area[pt[1]][pt[0]] = (0, 0, 255)

# corner points on contour
corner_line_points = []
for pt in corners_area_points:
    if contour[pt[1]][pt[0]] == 0.0:
         corner_line_points.append(pt)

# for pt in corner_line_points:
#     contour_rgb[pt[1]][pt[0]] = (0, 255, 0)

# merge points on corner lines
def merge_corner_lines_to_point(corner_line_points, contour_sorted):
    corner_points = []
    if corner_line_points is None or contour_sorted is None:
        return corner_points
    # merge point on corner line
    i = 0
    start_id = end_id = i
    while True:
        if i == len(contour_sorted)-1:
            break
        if contour_sorted[i] in corner_line_points:
            start_id = i
            end_id = i
            for j in range(i+1, len(contour_sorted)):
                if contour_sorted[j] in corner_line_points:
                    continue
                else:
                    end_id = j-1
                    break
            midd_id = start_id + int((end_id-start_id)/2.)
            corner_points.append(contour_sorted[midd_id])
            i = end_id

        i += 1

    return corner_points

corner_all_points = merge_corner_lines_to_point(corner_line_points, contour_sorted)

# for pt in corner_all_points:
#     contour_rgb[pt[1]][pt[0]] = (255, 0, 0)

# valid corner points should be close to the cross points
corner_points = []
threshold_distance = 40
for pt in corner_all_points:
    dist_cross = min_distance_point2pointlist(pt, cross_points)
    dist_end = min_distance_point2pointlist(pt, end_points)
    if dist_cross < threshold_distance and dist_end > threshold_distance / 3.:
        corner_points.append(pt)

for pt in corner_points:
    contour_rgb[pt[1]][pt[0]] = (0, 0, 255)
print("corner points num: %d" % len(corner_points))

# cluster the corner points
dist_threshold = 30
corner_points_cluster = []
used_index = []
for i in range(len(cross_points)):
    cross_pt = cross_points[i]
    cluster = []
    for j in range(len(corner_points)):
        if j in used_index:
            continue
        corner_pt = corner_points[j]
        dist = math.sqrt((cross_pt[0]-corner_pt[0])**2 + (cross_pt[1]-corner_pt[1])**2)
        if dist < dist_threshold:
            cluster.append(corner_pt)
            used_index.append(j)
    if cluster:
        corner_points_cluster.append(cluster)
print("corner cluster num:%d" % len(corner_points_cluster))
print(corner_points_cluster)

# detect corner points type: two point, four point (rectangle or diamond)


crop_lines = []
for i in range(len(corner_points_cluster)):
    corner_clt = corner_points_cluster[i]
    if len(corner_clt) == 2:
        print(" tow points")
        crop_lines.append((corner_clt))
    elif len(corner_clt) == 4:
        # rectangle or diamond (vertical/horizon or pie/na)
        min_offset = 1000
        for i in range(len(corner_clt)):
            pt1 = corner_clt[i]
            if i == len(corner_clt) - 1:
                pt2 = corner_clt[0]
            else:
                pt2 = corner_clt[i+1]
            offset = abs(pt1[0]-pt2[0])
            if offset <= min_offset:
                min_offset = offset
        if min_offset <= 10:
            print("rectangle")
            if abs(corner_clt[0][0]-corner_clt[1][0]) <= 10:
                crop_lines.append((corner_clt[0], corner_clt[1]))
                crop_lines.append((corner_clt[2], corner_clt[3]))
                if abs(corner_clt[0][1] - corner_clt[2][1]) <= 10:
                    crop_lines.append((corner_clt[0], corner_clt[2]))
                    crop_lines.append((corner_clt[1], corner_clt[3]))
                else:
                    crop_lines.append((corner_clt[0], corner_clt[3]))
                    crop_lines.append((corner_clt[1], corner_clt[2]))
            elif abs(corner_clt[0][0] - corner_clt[2][0]) <= 10:
                crop_lines.append((corner_clt[0], corner_clt[2]))
                crop_lines.append((corner_clt[1], corner_clt[3]))

                if abs(corner_clt[0][1] - corner_clt[1][1]) <= 10:
                    crop_lines.append((corner_clt[0], corner_clt[1]))
                    crop_lines.append((corner_clt[2], corner_clt[3]))
                else:
                    crop_lines.append((corner_clt[0], corner_clt[3]))
                    crop_lines.append((corner_clt[1], corner_clt[2]))

        else:
            print("diamond")
            """
                                P3
                        P0              P2
                                P1
            """
            P0 = P1 = P2 = P3 = None
            min_x = min_y = 10000000
            max_x = max_y = 0
            for pt in corner_clt:
                if pt[0] > max_x:
                    max_x = pt[0]
                if pt[0] < min_x:
                    min_x = pt[0]
                if pt[1] > max_y:
                    max_y = pt[1]
                if pt[1] < min_y:
                    min_y = pt1[1]
            print("minx:%d miny:%d maxx:%d maxy:%d" % (min_x, min_y, max_x, max_y))

            for pt in corner_clt:
                if pt[0] == min_x:
                    P0 = pt
                elif pt[0] == max_x:
                    P2 = pt
                if pt[1] == min_y:
                    P3 = pt
                elif pt[1] == max_y:
                    P1 = pt
            crop_lines.append((P0, P1))
            crop_lines.append((P1, P2))
            crop_lines.append((P2, P3))
            crop_lines.append((P3, P0))

# display crop lines
crop_lines_points = []
for line in crop_lines:
    cv2.line(contour_rgb, line[0], line[1], (0, 255, 0), 1)
    next_pt = line[0]
    line_points = [line[0]]
#     while True:
#
#         if next_pt == line[1]:
#             break
#         x = next_pt[0]; y = next_pt[1]
#
#         """
#                 9 | 2 | 3
#                 8 |   | 4
#                 7 | 6 | 5
#         """
#
#         # p2
#         if contour_rgb[y-1][x][0] == 0 and contour_rgb[y-1][x][1] == 255 and contour_rgb[y-1][x][2] == 0 and (x, y-1) not in line_points:
#             next_pt = (x, y-1)
#             print("p2")
#         # p3
#         elif contour_rgb[y-1][x+1][0] == 0 and contour_rgb[y-1][x+1][1] == 255 and contour_rgb[y-1][x+1][2] == 0 and (x+1, y-1) not in line_points:
#             next_pt = (x+1, y-1)
#             print("p3")
#         # p4
#         elif contour_rgb[y][x+1][0] == 0 and contour_rgb[y][x+1][1] == 255 and contour_rgb[y][x+1][2] == 0 and (x+1, y) not in line_points:
#             next_pt = (x+1, y)
#             print("p4")
#         # p5
#         elif contour_rgb[y+1][x+1][0] == 0 and contour_rgb[y+1][x+1][1] == 255 and contour_rgb[y+1][x+1][2] == 0 and (x+1, y+1) not in line_points:
#             next_pt = (x+1, y+1)
#             print("p5")
#         # p6
#         elif contour_rgb[y+1][x][0] == 0 and contour_rgb[y+1][x][1] == 255 and contour_rgb[y+1][x][2] == 0 and (x, y+1) not in line_points:
#             next_pt = (x, y+1)
#             print("p6")
#         # p7
#         elif contour_rgb[y+1][x-1][0] == 0 and contour_rgb[y+1][x-1][1] == 255 and contour_rgb[y+1][x-1][2] == 0 and (x-1, y+1) not in line_points:
#             next_pt = (x-1, y+1)
#             print("p7")
#         # p8
#         elif contour_rgb[y][x-1][0] == 0 and contour_rgb[y][x-1][1] == 255 and contour_rgb[y][x-1][2] == 0 and (x-1, y) not in line_points:
#             next_pt = (x-1, y)
#             print("p8")
#         # p9
#         elif contour_rgb[y-1][x-1][0] == 0 and contour_rgb[y-1][x-1][1] == 255 and contour_rgb[y-1][x-1][2] == 0 and (x-1, y-1) not in line_points:
#             next_pt = (x-1, y-1)
#             print("p2")
#         line_points.append(next_pt)
#     line_points.append(line[1])
#
#     crop_lines_points.append(line_points)
#
# print("crop lines points num: %d" % len(crop_lines_points))

# crop character
print("contor point num: %d" % len(contour_sorted))
sub_contours = segmentContourBasedOnCornerPoints(contour_sorted, corner_points)
print("sub contours num: %d" % len(sub_contours))

# separate single region to several region
contour_separate_region = cv2.cvtColor(contour_rgb, cv2.COLOR_RGB2GRAY)
_, contour_separate_region = cv2.threshold(contour_separate_region, 240, 255, cv2.THRESH_BINARY)






# cv2.imshow("radicals", radicals)
cv2.imshow("contour", contour)
# cv2.imshow("skeleton", skeleton)
cv2.imshow("skeleton rgb", skeleton_rgb)
# cv2.imshow("img_corner_area", img_corner_area)
cv2.imshow("contour_rgb", contour_rgb)
cv2.imshow("contour_separate_region", contour_separate_region)

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






