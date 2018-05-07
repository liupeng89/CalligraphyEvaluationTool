import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeBranchOfSkeleton, getEndPointsOfSkeletonLine, \
                          getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints, merge_corner_lines_to_point

# 1133壬 2252支 0631叟
path = "0631叟.jpg"

img = cv2.imread(path)
img_rgb = img.copy()

# get contour of image
contour = getContourOfImage(img)
contour = getSkeletonOfImage(contour)
contour = np.array(contour, dtype=np.uint8)

contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)

contour_points_sorted = sortPointsOnContourOfImage(contour)
print("contour points num:%d" % len(contour_points_sorted))

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(img.shape)
# get skeleton
skeleton = getSkeletonOfImage(img)
skeleton = removeBranchOfSkeleton(skeleton, distance_threshod=20)
skeleton = np.array(skeleton, dtype=np.uint8)
skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)

# end points
end_points = getEndPointsOfSkeletonLine(skeleton)
cross_points = getCrossPointsOfSkeletonLine(skeleton)


# corner area detect
img_corner = np.float32(img.copy())
dst = cv2.cornerHarris(img_corner, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

corners_area_points = []
for y in range(dst.shape[0]):
    for x in range(dst.shape[1]):
        if dst[y][x] > 0.1 * dst.max():
            corners_area_points.append((x, y))

img_corner_area = img_rgb.copy()

for pt in corners_area_points:
    img_corner_area[pt[1]][pt[0]] = (0, 0, 255)
# corner lines in contour rgb
corners_line_points = []
for pt in corners_area_points:
    if pt in contour_points_sorted:
        corners_line_points.append(pt)
for pt in corners_line_points:
    contour_rgb[pt[1]][pt[0]] = (0, 255, 0)
# merge all points to single point in corner lines
corners_all_points = merge_corner_lines_to_point(corners_line_points, contour_points_sorted)

for pt in corners_all_points:
    contour_rgb[pt[1]][pt[0]] = (255, 0, 0)

# obtain all valid corner points
corners_points = []
threshold_distance = 25
for pt in corners_all_points:
    dist_cross = min_distance_point2pointlist(pt, cross_points)
    dist_end = min_distance_point2pointlist(pt, end_points)
    if dist_cross < threshold_distance and dist_end > threshold_distance / 3.:
        corners_points.append(pt)

for pt in corners_points:
    contour_rgb[pt[1]][pt[0]] = (0, 0, 255)
print("corner points num: %d" % len(corners_points))


# cv2.imshow("img", img)
# cv2.imshow("contour", contour)
cv2.imshow("skeleton", skeleton)
cv2.imshow("img_corner_area", img_corner_area)
cv2.imshow("contour_rgb", contour_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
