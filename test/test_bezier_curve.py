# import numpy as np
# from scipy.misc import comb
#
# def bernstein_poly(i, n, t):
#     """
#     The Bernstein polynomial of n, i as a function of t
#     :param i:
#     :param n:
#     :param t:
#     :return:
#     """
#     return comb(n,i) * (t**(n-i)) * (1 - t) ** i
#
#
# def bezier_curve(points, nTimes=1000):
#
#     nPoints = len(points)
#     xPoints = np.array([p[0] for p in points])
#     yPoints = np.array([p[1] for p in points])
#
#     t = np.linspace(0.0, 1.0, nTimes)
#
#     polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
#
#     xvals = np.dot(xPoints, polynomial_array)
#     yvals = np.dot(yPoints, polynomial_array)
#
#     return xvals, yvals
#
# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#
#     nPoints = 4
#     points = np.random.rand(nPoints,2)*200
#     xpoints = [p[0] for p in points]
#     ypoints = [p[1] for p in points]
#
#     xvals, yvals = bezier_curve(points, nTimes=1000)
#     plt.plot(xvals, yvals)
#     plt.plot(xpoints, ypoints, "ro")
#     for nr in range(len(points)):
#         plt.text(points[nr][0], points[nr][1], nr)
#
#     plt.show()
import cv2
import numpy as np
import math

from utils.Functions import createBlankGrayscaleImage, getAllMiniBoundingBoxesOfImage, getContourOfImage, \
                            getConnectedComponents, removeBreakPointsOfContour, sortPointsOnContourOfImage, \
                            fitCurve, draw_cubic_bezier, createBlankRGBImage, getSkeletonOfImage, \
                            removeBranchOfSkeletonLine, getEndPointsOfSkeletonLine, getCrossPointsOfSkeletonLine

path = "test_images/quan1.png"

# 1. Load image
img = cv2.imread(path)
img_gray = None

# 2. RGB image -> grayscale and bitmap
if len(img.shape) == 3:
    # rgb image
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

elif len(img.shape) == 2:
    # grayscale img
    img_gray = img

_, img_bit = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

components = getConnectedComponents(img_bit)

# for i in range(len(components)):
#     cv2.imshow("c_%d" % i, components[i])


component = components[1]

skeleton = getSkeletonOfImage(component)

corner_component = np.float32(component)

dst = cv2.cornerHarris(corner_component, blockSize=3, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

corners_area_points = []
for y in range(dst.shape[0]):
    for x in range(dst.shape[1]):
        if dst[y][x] > 0.1 * dst.max():
            corners_area_points.append((x, y))
print("corner area points num: %d" % len(corners_area_points))

# 6. Determine center points of corner areas
blank_gray = createBlankGrayscaleImage(component)
for pt in corners_area_points:
    blank_gray[pt[1]][pt[0]] = 0.0

rectangles = getAllMiniBoundingBoxesOfImage(blank_gray)

corners_area_center_points = []
for rect in rectangles:
    corners_area_center_points.append((rect[0] + int(rect[2] / 2.), rect[1] + int(rect[3] / 2.)))
print("corner area center points num: %d" % len(corners_area_center_points))

end_points = getEndPointsOfSkeletonLine(skeleton)
cross_points = getCrossPointsOfSkeletonLine(skeleton)

valid_corners_area_center_points = []
dist_threshold = 40
for pt in corners_area_center_points:
    is_valid = False
    for ept in end_points:
        dist = math.sqrt((pt[0] - ept[0]) ** 2 + (pt[1] + ept[1]) ** 2)
        if dist <= dist_threshold:
            is_valid = True
            break
    if is_valid:
        valid_corners_area_center_points.append(pt)
        continue
    for cpt in cross_points:
        dist = math.sqrt((pt[0] - cpt[0]) ** 2 + (pt[1] - cpt[1]) ** 2)
        if dist <= dist_threshold:
            is_valid = True
            break
    if is_valid:
        valid_corners_area_center_points.append(pt)

print("valid corner area center points num: %d" % len(valid_corners_area_center_points))

del blank_gray

# 7. Get all contours of component
component_contours = getContourOfImage(component)
contours = getConnectedComponents(component_contours, connectivity=8)
print("contours num: %d" % len(contours))

# 8. Process contours to get closed and 1-pixel width contours
contours_processed = []
for cont in contours:
    cont = removeBreakPointsOfContour(cont)
    contours_processed.append(cont)
print("contours processed num: %d" % len(contours_processed))

# 9. Find corner points of conthours closed to corner region center points. For each contour, there is a coner points list.
contours_corner_points = []
for i in range(len(contours_processed)):
    corner_points = []
    contour = contours_processed[i]

    for pt in valid_corners_area_center_points:
        x0 = target_x = pt[0];
        y0 = target_y = pt[1]
        min_dist = 10000
        # search target point in region: 20 * 20 of center is (x0, y0)
        for y in range(y0 - 10, y0 + 10):
            for x in range(x0 - 10, x0 + 10):
                if contour[y][x] == 255:
                    continue
                dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                if dist < min_dist:
                    min_dist = dist;
                    target_x = x;
                    target_y = y
        if min_dist < 5:
            corner_points.append((target_x, target_y))

    contours_corner_points.append(corner_points)
total_num = 0
for cont in contours_corner_points:
    total_num += len(cont)
if total_num == len(valid_corners_area_center_points):
    print("corner points not ignored")
else:
    print("corner points be ignored")

contour = contours_processed[0]
contour_points = contours_corner_points[0]

contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
for pt in contour_points:
    contour_rgb[pt[1]][pt[0]] = (0, 255, 0)

cv2.imshow("c", component)
cv2.imshow("skleton", skeleton)

cv2.imshow("contour_rgb", contour_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
