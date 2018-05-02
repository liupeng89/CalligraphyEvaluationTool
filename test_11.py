# import numpy as np
# import cv2
#
# from utils.Functions import getContourOfImage, sortPointsOnContourOfImage, getSingleMaxBoundingBoxOfImage
#
# from math import acos
# from math import sqrt
# from math import pi
#
# import math
#
# def length(v):
#     return sqrt(v[0]**2+v[1]**2)
# def dot_product(v,w):
#    return v[0]*w[0]+v[1]*w[1]
# def determinant(v,w):
#    return v[0]*w[1]-v[1]*w[0]
# def inner_angle(v,w):
#    cosx=dot_product(v,w)/(length(v)*length(w))
#    if cosx > 1.0:
#        cosx = 1.
#    elif cosx < -1.0:
#        cosx = -1.
#    # print(cosx)
#    rad=acos(cosx) # in radians
#    return rad*180/pi # returns degrees
# def angle_clockwise(A, B):
#     inner=inner_angle(A,B)
#     det = determinant(A,B)
#     if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
#         return inner
#     else: # if the det > 0 then A is immediately clockwise of B
#         return 360-inner
#
# def get_angle(p0, p1, p2):
#     v0 = np.array(p1) - np.array(p0)
#     v1 = np.array(p2) - np.array(p1)
#
#     angle = math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
#     return np.degrees(angle)
#
# def get_angle1(p0, p1, p2):
#     v0 = np.array(p1) - np.array(p0)
#     v1 = np.array(p2) - np.array(p1)
#
#     angle = math.atan2(v0[0]*v1[1]-v0[1]*v1[0], v0[0]*v1[0]+v0[1]*v1[1])
#     return np.degrees(angle)


# a = np.array([1, 2])
# b = np.array([1, 1])
# c = np.array([2, 1])
#
# print(get_angle(a, b, c))
#
# a = np.array([2, 2])
# b = np.array([2, 1])
# c = np.array([1, 1])
#
# print(get_angle(a, b, c))
#
# a1 = np.array([1, 3])
# a2 = np.array([2, 3])
# a3 = np.array([2, 2])
# a4 = np.array([3, 2])
# a5 = np.array([3, 1])
# a6 = np.array([2, 1])
# a7 = np.array([2, 0])
# a8 = np.array([1, 0])
# a9 = np.array([1, 1])
# a10 = np.array([0, 1])
# a11 = np.array([0, 2])
# a12 = np.array([1, 2])
#
# print(get_angle(a1, a2, a3))
# print(get_angle(a2, a3, a4))
# print(get_angle(a3, a4, a5))
# print(get_angle(a4, a5, a6))
# print(get_angle(a5, a6, a7))
# print(get_angle(a6, a7, a8))
# print(get_angle(a7, a8, a9))
# print(get_angle(a8, a9, a10))
# print(get_angle(a9, a10, a11))
# print(get_angle(a10, a11, a12))
#
# print("-----------------")
#
# print(get_angle1(a1, a2, a3))
# print(get_angle1(a2, a3, a4))
# print(get_angle1(a3, a4, a5))
# print(get_angle1(a4, a5, a6))
# print(get_angle1(a5, a6, a7))
# print(get_angle1(a6, a7, a8))
# print(get_angle1(a7, a8, a9))
# print(get_angle1(a8, a9, a10))
# print(get_angle1(a9, a10, a11))
# print(get_angle1(a10, a11, a12))






# print(angle_clockwise(A, B))

# img_path = "0001ding.jpg"
#
# img = cv2.imread(img_path, 0)
# _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# contour = getContourOfImage(img)
# contour_sorted = sortPointsOnContourOfImage(contour)
#
# obtuse_points = []
#
# for i in range(len(contour_sorted)):
#     start_index = i
#     middl_index = i + 10
#     end_index = i + 20
#
#     if end_index >= len(contour_sorted)-1:
#         break
#     angle = get_angle(contour_sorted[start_index], contour_sorted[middl_index], contour_sorted[end_index])
#     # vect_a = np.array(contour_sorted[middl_index]) - np.array(contour_sorted[start_index])
#     # vect_b = np.array(contour_sorted[end_index]) - np.array(contour_sorted[middl_index])
#     #
#     # angle = angle_clockwise(vect_a, vect_b)
#     print(angle)
#
#     if angle < 0.0:
#         obtuse_points.append(np.array(contour_sorted[middl_index]))
#     # break
#
# print("obtuse len: %d" % len(obtuse_points))
#
# contour_rgb = cv2.cvtColor(np.array(contour, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
#
# print(contour_sorted[0])
#
# contour_rgb[contour_sorted[0][1]][contour_sorted[0][0]] = (0, 255, 0)
#
# for pt in obtuse_points:
#     contour_rgb[pt[1]][pt[0]] = (0, 0, 255)
#
# cv2.imshow("contour", contour)0554十.jpg
# cv2.imshow("contour rgb", contour_rgb)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import numpy as np
import cv2
import math
from skimage.morphology import skeletonize


from utils.Functions import getContourOfImage, sortPointsOnContourOfImage, removeBreakPointsOfContour, \
                            getSkeletonOfImage, removeBranchOfSkeletonLine, getEndPointsOfSkeletonLine, \
                            getCrossPointsOfSkeletonLine, getNumberOfValidPixels

# 0107亻  1133壬  0554十
path = "1133壬.jpg"

img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_copy = img.copy()

contour = getContourOfImage(img)
contour = removeBreakPointsOfContour(contour)

contour_sorted = sortPointsOnContourOfImage(contour)
print(len(contour_sorted))
print(contour_sorted[0])
print(contour_sorted[-1])

img = np.float32(img)
dst = cv2.cornerHarris(img, 3, 3, 0.05)
dst = cv2.dilate(dst, None)

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
contour_rgb1 = contour_rgb.copy()
print(dst.shape)

corner_points = []
for y in range(dst.shape[0]):
    for x in range(dst.shape[1]):
        if dst[y][x] > 0.1*dst.max():
            corner_points.append((x, y))
print("corner points len: %d" % len(corner_points))

for pt in corner_points:
    if img[pt[1]][pt[0]] == 0:
        img_rgb[pt[1]][pt[0]] = (0, 255, 0)
    else:
        img_rgb[pt[1]][pt[0]] = (0, 0, 255)

corners_nomerged = []
for y in range(contour.shape[0]):
    for x in range(contour.shape[1]):
        if contour[y][x] == 0 and (x, y) in corner_points:
            contour_rgb[y][x] = (0, 0, 255)
            corners_nomerged.append((x, y))
print("corners sequence len: %d" % len(corners_nomerged))

# merge points of corner points
corners_segments = []
corners_merged = []
i = 0
while True:
    midd_index = -1
    pt = contour_sorted[i]
    if pt in corners_nomerged:
        # red point
        start = i
        end = start
        while True:
            end += 1
            if end >= len(contour_sorted):
                break
            # next point
            next_pt = contour_sorted[end]
            if next_pt in corners_nomerged:
                # red point
                continue
            else:
                # black point
                break
        end -= 1
        midd_index = start + int((end-start)/2.0)
        i = end

    i += 1

    if i >= len(contour_sorted):
        break
    if midd_index != -1:
        corners_merged.append(contour_sorted[midd_index])

for pt in corners_merged:
    contour_rgb[pt[1]][pt[0]] = (255, 0, 0)

print("cornet segments: %d" % len(corners_segments))
print("corner merged len: %d" % len(corners_merged))

# get skeleton of image
skeleton = getSkeletonOfImage(img_copy)
skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)

end_points = getEndPointsOfSkeletonLine(skeleton)
cross_points = getCrossPointsOfSkeletonLine(skeleton)
print("end points len:%d" % len(end_points))
print("cross points len: %d" % len(cross_points))

# skeleton = removeBranchOfSkeletonLine(skeleton, end_points, cross_points, DIST_THRESHOLD=50)


for pt in cross_points:
    skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)
for pt in end_points:
    skeleton_rgb[pt[1]][pt[0]] = (255, 0, 0)

# remove points not be the C points
if cross_points is None or len(cross_points) == 0:
    print("single stroke, need not to extract!")
else:
    print("multi-strokes composed!")

def min_distance_point2pointlist(point, points):
    min_dist = 1000000000
    if point is None or points is None:
        return min_dist
    for pt in points:
        dist = math.sqrt((point[0]-pt[0])**2+(point[1]-pt[1])**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


singular_points = []
DIST_THRESHOLD = 30
for pt in corners_merged:
    dist = min_distance_point2pointlist(pt, cross_points)
    if dist < DIST_THRESHOLD:
        # this point is the singlular point on contour
        singular_points.append(pt)
print("singular points num: %d" % len(singular_points))

for pt in singular_points:
    contour_rgb[pt[1]][pt[0]] = (0, 255, 0)

# segment contour to sub-contours with singular points
sub_contour_index = []
for pt in singular_points:
    index = contour_sorted.index(pt)
    sub_contour_index.append(index)
print("sub contour index num: %d" % len(sub_contour_index))
print(sub_contour_index)

sub_contours = []
for i in range(len(sub_contour_index)):
    if i == len(sub_contour_index)-1:
        sub_contour = contour_sorted[sub_contour_index[i]:len(contour_sorted)] + contour_sorted[0: sub_contour_index[0]+1]
    else:
        sub_contour = contour_sorted[sub_contour_index[i]:sub_contour_index[i+1]+1]
    sub_contours.append(sub_contour)
print("sub contours num: %d" % len(sub_contours))

for i in range(len(sub_contours)):
    if i % 3 == 0:
        color = (0, 0, 255)
    elif i % 3 == 1:
        color = (0, 255, 0)
    elif i % 3 == 2:
        color = (255, 0, 0)
    for pt in sub_contours[i]:
        contour_rgb1[pt[1]][pt[0]] = color

# merge two sub-contours
# 亻T type
start_pt = sub_contours[0][-1]
end_pt = sub_contours[0][0]

stroke_img = np.ones_like(contour) * 255
stroke_img = np.array(stroke_img, dtype=np.uint8)

for pt in sub_contours[0]:
    stroke_img[pt[1]][pt[0]] = 0

cv2.line(stroke_img, start_pt, end_pt, 0, 1)
stroke_contour_sort = sortPointsOnContourOfImage(stroke_img)
stroke_contour_sort = np.array([stroke_contour_sort], "int32")
cv2.fillPoly(stroke_img, stroke_contour_sort, 0)










cv2.imshow("contour", contour)
cv2.imshow("img rgb", img_rgb)
cv2.imshow("contour rbg", contour_rgb)
cv2.imshow("sub contours", contour_rgb1)
cv2.imshow("stroke image", stroke_img)
# cv2.imshow("skeleton", skeleton)
cv2.imshow("skeleton rgb", skeleton_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()







# cv2.imshow("dst", img_rgb)


# contour = getContourOfImage(img)
#
# contour_sort_points = sortPointsOnContourOfImage(contour, isClockwise=False)
# print("contour points len: %d" % len(contour_sort_points))
#
# contour_len = len(contour_sort_points)
#
# theta_angles = []
# for i in range(1, contour_len):
#     pt1 = contour_sort_points[i-1]
#     pt2 = contour_sort_points[i]
#
#     theta = math.atan2((pt2[1]-pt1[1]), -(pt2[0]-pt1[0]))
#     theta_angles.append(theta)
#
# # N point and the first point
# theta = math.atan2(-(contour_sort_points[0][1]-contour_sort_points[contour_len-1][1]), (contour_sort_points[0][0]-contour_sort_points[contour_len-1][0]))
# theta_angles.append(theta)
#
# print("theta angles: %d" % len(theta_angles))
# print(theta_angles)
#
# single_points = []
# for i in range(len(theta_angles)):
#     if i+10 >= len(theta_angles):
#         break
#     delt = abs(theta_angles[i] - theta_angles[i+10])
#     if delt > math.pi:
#         single_points.append(contour_sort_points[i])
#
#
# contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
#
# for pt in single_points:
#     contour_rgb[pt[1]][pt[0]] = (0, 0, 255)