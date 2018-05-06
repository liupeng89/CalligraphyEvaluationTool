import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeBranchOfSkeleton, getEndPointsOfSkeletonLine, \
                          getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints

# 1133壬 2252支
path = "1133壬.jpg"

img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

components = getConnectedComponents(img)

print("radicals num: %d" % len(components))

radicals = components[0]
radicals = np.array(radicals, dtype=np.uint8)

# # contour
contour = getContourOfImage(radicals)
contour = np.array(contour, dtype=np.uint8)


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
# for pt in end_points:
#     skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)
# for pt in cross_points:
#     skeleton_rgb[pt[1]][pt[0]] = (0, 255, 0)

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

for pt in corner_line_points:
    contour_rgb[pt[1]][pt[0]] = (0, 255, 0)

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

for pt in corner_all_points:
    contour_rgb[pt[1]][pt[0]] = (255, 0, 0)

# valid corner points should be close to the cross points
corner_points = []
threshold_distance = 30
for pt in corner_all_points:
    dist_cross = min_distance_point2pointlist(pt, cross_points)
    dist_end = min_distance_point2pointlist(pt, end_points)
    if dist_cross < threshold_distance and dist_end > threshold_distance / 3.:
        corner_points.append(pt)

for pt in corner_points:
    contour_rgb[pt[1]][pt[0]] = (0, 0, 255)
print("corner points num: %d" % len(corner_points))

# cluster the corner points
dist_threshold = 40
corner_points_cluster = []
used_index = []
for i in range(len(corner_points)):
    if i in used_index:
        continue
    pt1 = corner_points[i]
    cluster = [pt1]
    used_index.append(i)
    for j in range(len(corner_points)):
        if i == j or j in used_index:
            continue
        pt2 = corner_points[j]
        dist = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
        if dist <= dist_threshold:
            cluster.append(pt2)
            used_index.append(j)

    corner_points_cluster.append(cluster)

print("clust num: %d" % len(corner_points_cluster))

# segment contour to sub-contour
print("contor point num: %d" % len(contour_sorted))
sub_contours = segmentContourBasedOnCornerPoints(contour_sorted, corner_points)
print("sub contour num: %d" % len(sub_contours))

# corner points correspondence

def isInOneSubContour(pt1, pt2, sub_contours):
    if pt1 is None or pt2 is None or sub_contours is None:
        return False
    label = False
    for sub in sub_contours:
        if pt1 in sub and pt2 in sub:
            label = True
            break
    return label

# co-linear  |y1-y2| <= 10 pixels and not in same sub-contour
co_linear_points = []
parallel_points = []
co_sub_contour = []
for i in range(len(corner_points)):
    pt1 = corner_points[i]
    for j in range(len(corner_points)):
        if i == j:
            continue
        pt2 = corner_points[j]

        # co-linear
        if abs(pt1[1] - pt2[1]) <= 10 and not isInOneSubContour(pt1, pt2, sub_contours):
            # co-linear
            pair = set((pt1, pt2))
            if pair not in co_linear_points:
                co_linear_points.append(pair)

        # parallel
        if abs(pt1[0] - pt2[0]) <= 10 and not isInOneSubContour(pt1, pt2, sub_contours):
            # parallel
            pair = set((pt1, pt2))
            if pair not in parallel_points and pair not in co_linear_points:
                parallel_points.append(pair)

        # co sub-contour
        if isInOneSubContour(pt1, pt2, sub_contours):
            # co subcontour
            if [pt1, pt2] not in parallel_points and [pt2, pt1] not in parallel_points:
                co_sub_contour.append([pt1, pt2])

print(co_linear_points)
print(parallel_points)
print(co_sub_contour)








# cv2.imshow("radicals", radicals)
cv2.imshow("contour", contour)
# cv2.imshow("skeleton", skeleton)
# cv2.imshow("skeleton rgb", skeleton_rgb)
# cv2.imshow("img_corner_area", img_corner_area)
# cv2.imshow("contour_rgb", contour_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()





