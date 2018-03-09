# # import numpy as np
# # import cv2
# # from matplotlib import pyplot as plt
# #
# # from functions.AddBoundingBox import addBoundingBox
# #
# # src_path = "../characters/src_dan_processed.png"
# # tag_path = "../characters/tag_dan_processed.png"
# #
# # src_img = cv2.imread(src_path, 0)
# # tag_img = cv2.imread(tag_path, 0)
# #
# # src_minx, src_miny, src_minw, src_minh = addBoundingBox(src_img)
# # tag_minx, tag_miny, tag_minw, tag_minh = addBoundingBox(tag_img)
# #
# # src_min_bounding = src_img[src_miny: src_miny+src_minh, src_minx: src_minx+src_minw]
# # tag_min_bounding = tag_img[tag_miny: tag_miny+tag_minh, tag_minx: tag_minx+tag_minw]
# #
# # src_maxw = max(src_minw, src_minh)
# # tag_maxw = max(tag_minw, tag_minh)
# #
# # src_new_square = np.ones((src_maxw, src_maxw)) * 255
# # tag_new_square = np.ones((tag_maxw, tag_maxw)) * 255
# #
# # # new src square
# # for y in range(src_min_bounding.shape[0]):
# #     for x in range(src_min_bounding.shape[1]):
# #         if src_min_bounding.shape[0] > src_min_bounding.shape[1]:
# #             # height > width
# #             offset = int((src_min_bounding.shape[0] - src_min_bounding.shape[1]) / 2)
# #             src_new_square[y][x+offset] = src_min_bounding[y][x]
# #         else:
# #             # height < width
# #             offset = int((src_min_bounding.shape[1] - src_min_bounding.shape[0]) / 2)
# #             src_new_square[y+offset][x] = src_min_bounding[y][x]
# #
# #
# # # new tag square
# # for y in range(tag_min_bounding.shape[0]):
# #     for x in range(tag_min_bounding.shape[1]):
# #         if tag_min_bounding.shape[0] > tag_min_bounding.shape[1]:
# #             # height > width
# #             offset = int((tag_min_bounding.shape[0] - tag_min_bounding.shape[1]) / 2)
# #             tag_new_square[y][x+offset] = tag_min_bounding[y][x]
# #         else:
# #             # height < width
# #             offset = int((tag_min_bounding.shape[1] - tag_min_bounding.shape[0]) / 2)
# #             tag_new_square[y+offset][x] = tag_min_bounding[y][x]
# #
# # # resize new square to same size between the source image and target image
# # if src_new_square.shape[0] > tag_new_square.shape[0]:
# #     # src > tag
# #     src_new_square = cv2.resize(src_new_square, tag_new_square.shape)
# # else:
# #     # src < tag
# #     tag_new_square = cv2.resize(tag_new_square, src_new_square.shape)
# #
# # # histogram
# # # plt.hist(src_new_square.ravel(), 256, [0, 256]); plt.show()
# # # plt.hist(tag_new_square.ravel(), 256, [0, 256]); plt.show()
# #
# # # x-axis and y-axis statistics histogram
# # src_x_hist = np.zeros(src_new_square.shape[1])
# # src_y_hist = np.zeros(src_new_square.shape[0])
# #
# # tag_x_hist = np.zeros(tag_new_square.shape[1])
# # tag_y_hist = np.zeros(tag_new_square.shape[0])
# #
# # for y in range(src_new_square.shape[0]):
# #     for x in range(src_new_square.shape[1]):
# #         if src_new_square[y][x] == 0:
# #             src_y_hist[y] += 1
# #             src_x_hist[x] += 1
# #
# # for y in range(tag_new_square.shape[0]):
# #     for x in range(tag_new_square.shape[1]):
# #         if tag_new_square[y][x] == 0:
# #             tag_y_hist[y] += 1
# #             tag_x_hist[x] += 1
# #
# # print(src_x_hist)
# # print(src_y_hist)
# #
# # plt.subplot(221); plt.plot(src_x_hist)
# # plt.subplot(222); plt.plot(src_y_hist)
# # plt.subplot(223); plt.plot(tag_x_hist)
# # plt.subplot(224); plt.plot(tag_y_hist)
# #
# # plt.show()
# #
# #
# # # img_file = "../characters/tag_bing copy.png.png"
# # #
# # # img = cv2.imread(img_file, 0)
# # #
# # # print(img.shape)
# # #
# # # rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# # #
# # # WIDTH = img.shape[0]
# # # HEIGHT = img.shape[1]
# # #
# # #
# # #
# # # # moments
# # # im2, contours, hierarchy = cv2.findContours(img, 1, 2)
# # # print("Contours len: %s " % len(contours))
# # #
# # # cnt = contours[0]
# # # M = cv2.moments(cnt)
# # # # print(M)
# # #
# # # # center of mass
# # # cx = int(M['m10'] / M['m00'])
# # # cy = int(M['m01'] / M['m00'])
# # # print('( %d, %d)' % (cy, cx))
# # #
# # #
# # # # area
# # # area = cv2.contourArea(cnt)
# # #
# # # print("Area:", area)
# # #
# # # minx = WIDTH
# # # miny = HEIGHT
# # # maxx = 0
# # # maxy = 0
# # # # Bounding box
# # # for i in range(len(contours)):
# # #     x, y, w, h = cv2.boundingRect(contours[i])
# # #     if w > 0.95 * WIDTH and h > 0.95 * HEIGHT:
# # #         continue
# # #
# # #     if x < minx:
# # #         minx = x
# # #     if y < miny:
# # #         miny = y
# # #     if x+w > maxx:
# # #         maxx = x+w
# # #     if y+h > maxy:
# # #         maxy = y+h
# # #
# # #     cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (0,255,0), 2)
# # #
# # # cv2.rectangle(rgb_img, (minx, miny), (maxx, maxy), (255, 0, 0), 3)
# # #
# # # min_bound_width = maxx - minx + 1;
# # # min_bound_height = maxy - miny + 1;
# # #
# # # aspect_ratio = min_bound_height / min_bound_width * 1.0
# # # print("Aspect ratio: %f \n" % aspect_ratio)
# #
# #
# # # convex hull
# #
# #
# # # cv2.imshow("src", src_new_square)
# # # cv2.imshow("tag", tag_new_square)
# # #
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
#
# # a = [[0, 3, 4, 8], [1, 2, 5, 6], [2, 1, 6], [3, 0, 4, 8], [4, 0, 3, 8, 10], [5, 1, 6], [6, 1, 2, 5], [7, 9], [8, 0, 3, 4, 10], [9, 7], [10, 4, 8], [11], [12, 13, 14, 20], [13, 12, 14, 20], [14, 12, 13, 20, 28], [15, 21, 22, 26], [16, 18, 23, 30], [17, 19, 24, 25], [18, 16, 23, 27, 30], [19, 17, 24, 25], [20, 12, 13, 14, 28, 29], [21, 15, 22, 26], [22, 15, 21, 26], [23, 16, 18, 27, 30], [24, 17, 19, 25], [25, 17, 19, 24], [26, 15, 21, 22], [27, 18, 23, 30], [28, 14, 20, 29], [29, 20, 28], [30, 16, 18, 23, 27, 33], [31, 32, 34, 37, 39, 40], [32, 31, 37, 39, 40], [33, 30, 35, 38, 42], [34, 31, 37, 39, 40, 45], [35, 33, 38, 42], [36, 41, 43], [37, 31, 32, 34, 39, 40, 45], [38, 33, 35, 42, 46], [39, 31, 32, 34, 37, 40, 45], [40, 31, 32, 34, 37, 39, 45], [41, 36, 43], [42, 33, 35, 38, 46], [43, 36, 41], [44], [45, 34, 37, 39, 40], [46, 38, 42], [47, 52, 55], [48, 50], [49, 51], [50, 48, 56], [51, 49], [52, 47, 55], [53, 54], [54, 53], [55, 47, 52], [56, 50], [57, 58, 59, 62], [58, 57, 59, 62], [59, 57, 58, 62], [60, 61, 64], [61, 60, 64, 67], [62, 57, 58, 59, 66], [63, 65], [64, 60, 61, 67], [65, 63], [66, 62], [67, 61, 64], [68], [69, 70, 72, 75], [70, 69, 72, 75, 78], [71, 76], [72, 69, 70, 75, 78], [73, 74, 77], [74, 73, 77], [75, 69, 70, 72, 78], [76, 71], [77, 73, 74], [78, 70, 72, 75], [79], [80]]
# #
# # final_clustor = []
# # used_index = []
# # for i in range(len(a)):
# #     if i in used_index:
# #         continue
# #
# #     new_clustor = a[i]
# #
# #     for j in range(i+1, len(a)):
# #         if len(set(new_clustor).intersection(set(a[j]))) == 0:
# #             continue
# #         new_clustor = list(set(new_clustor).union(set(a[j])))
# #         used_index.append(j)
# #     final_clustor.append(new_clustor)
# #
# # print(final_clustor)
# #
# # a = [0, 3, 4, 8]
# # b = [1,2,5,6]
# # c = [8,0,3,4,10]
# #
# # print(list(set(a).intersection(set(b))))
# # print(set(a).intersection(set(c)))
#
#
# # import cv2
# # import numpy as np
# # from utils.Functions import getBoundingBoxes
# # import math
# #
# #
# # def main():
# #     pass
#
#
# # import numpy as np
# # from scipy.misc import comb
# #
# # def bernstein_poly(i, n, t):
# #     """
# #      The Bernstein polynomial of n, i as a function of t
# #     """
# #
# #     return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
# #
# #
# # def bezier_curve(points, nTimes=1000):
# #     """
# #        Given a set of control points, return the
# #        bezier curve defined by the control points.
# #
# #        points should be a list of lists, or list of tuples
# #        such as [ [1,1],
# #                  [2,3],
# #                  [4,5], ..[Xn, Yn] ]
# # #         nTimes is the number of time steps, defaults to 1000
# #
# #         See http://processingjs.nihongoresources.com/bezierinfo/
# #     """
#
# #     nPoints = len(points)
# #     xPoints = np.array([p[0] for p in points])
# #     yPoints = np.array([p[1] for p in points])
# #
# #     t = np.linspace(0.0, 1.0, nTimes)
# #
# #     polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
# #
# #     xvals = np.dot(xPoints, polynomial_array)
# #     yvals = np.dot(yPoints, polynomial_array)
# #
# #     return xvals, yvals
# #
# #
# # if __name__ == "__main__":
# #     from matplotlib import pyplot as plt
# #
# #     nPoints = 4
# #     points = np.random.rand(nPoints,2)*200
# #     xpoints = [p[0] for p in points]
# #     ypoints = [p[1] for p in points]
# #
# #     xvals, yvals = bezier_curve(points, nTimes=1000)
# #     plt.plot(xvals, yvals)
# #     plt.plot(xpoints, ypoints, "ro")
# #     for nr in range(len(points)):
# #         plt.text(points[nr][0], points[nr][1], nr)
# #
# #     plt.show()
#
# """
# Ported from Paper.js - The Swiss Army Knife of Vector Graphics Scripting.
# http://paperjs.org/
# Copyright (c) 2011 - 2014, Juerg Lehni & Jonathan Puckey
# http://scratchdisk.com/ & http://jonathanpuckey.com/
# Distributed under the MIT license. See LICENSE file for details.
# All rights reserved.
# An Algorithm for Automatically Fitting Digitized Curves
# by Philip J. Schneider
# from "Graphics Gems", Academic Press, 1990
# Modifications and optimisations of original algorithm by Juerg Lehni.
# Ported by Gumble, 2015.
# """
#
# import math
#
# TOLERANCE = 10e-6
# EPSILON = 10e-12
#
#
# class Point:
#     __slots__ = ['x', 'y']
#
#     def __init__(self, x, y=None):
#         if y is None:
#             self.x, self.y = x[0], x[1]
#         else:
#             self.x, self.y = x, y
#
#     def __repr__(self):
#         return 'Point(%r, %r)' % (self.x, self.y)
#
#     def __str__(self):
#         return '%G,%G' % (self.x, self.y)
#
#     def __complex__(self):
#         return complex(self.x, self.y)
#
#     def __hash__(self):
#         return hash(self.__complex__())
#
#     def __bool__(self):
#         return bool(self.x or self.y)
#
#     def __add__(self, other):
#         if isinstance(other, Point):
#             return Point(self.x + other.x, self.y + other.y)
#         else:
#             return Point(self.x + other, self.y + other)
#
#     def __sub__(self, other):
#         if isinstance(other, Point):
#             return Point(self.x - other.x, self.y - other.y)
#         else:
#             return Point(self.x - other, self.y - other)
#
#     def __mul__(self, other):
#         if isinstance(other, Point):
#             return Point(self.x * other.x, self.y * other.y)
#         else:
#             return Point(self.x * other, self.y * other)
#
#     def __truediv__(self, other):
#         if isinstance(other, Point):
#             return Point(self.x / other.x, self.y / other.y)
#         else:
#             return Point(self.x / other, self.y / other)
#
#     def __neg__(self):
#         return Point(-self.x, -self.y)
#
#     def __len__(self):
#         return math.hypot(self.x, self.y)
#
#     def __eq__(self, other):
#         try:
#             return self.x == other.x and self.y == other.y
#         except Exception:
#             return False
#
#     def __ne__(self, other):
#         try:
#             return self.x != other.x or self.y != other.y
#         except Exception:
#             return True
#
#     add = __add__
#     subtract = __sub__
#     multiply = __mul__
#     divide = __truediv__
#     negate = __neg__
#     getLength = __len__
#     equals = __eq__
#
#     def copy(self):
#         return Point(self.x, self.y)
#
#     def dot(self, other):
#         return self.x * other.x + self.y * other.y
#
#     def normalize(self, length=1):
#         current = self.__len__()
#         scale = length / current if current != 0 else 0
#         return Point(self.x * scale, self.y * scale)
#
#     def getDistance(self, other):
#         return math.hypot(self.x - other.x, self.y - other.y)
#
#
# class Segment:
#
#     def __init__(self, *args):
#         self.point = Point(0, 0)
#         self.handleIn = Point(0, 0)
#         self.handleOut = Point(0, 0)
#         if len(args) == 1:
#             if isinstance(args[0], Segment):
#                 self.point = args[0].point
#                 self.handleIn = args[0].handleIn
#                 self.handleOut = args[0].handleOut
#             else:
#                 self.point = args[0]
#         elif len(args) == 2 and isinstance(args[0], (int, float)):
#             self.point = Point(*args)
#         elif len(args) == 2:
#             self.point = args[0]
#             self.handleIn = args[1]
#         elif len(args) == 3:
#             self.point = args[0]
#             self.handleIn = args[1]
#             self.handleOut = args[2]
#         else:
#             self.point = Point(args[0], args[1])
#             self.handleIn = Point(args[2], args[3])
#             self.handleOut = Point(args[4], args[5])
#
#     def __repr__(self):
#         return 'Segment(%r, %r, %r)' % (self.point, self.handleIn, self.handleOut)
#
#     def __hash__(self):
#         return hash((self.point, self.handleIn, self.handleOut))
#
#     def __bool__(self):
#         return bool(self.point or self.handleIn or self.handleOut)
#
#     def getPoint(self):
#         return self.point
#
#     def setPoint(self, other):
#         self.point = other
#
#     def getHandleIn(self):
#         return self.handleIn
#
#     def setHandleIn(self, other):
#         self.handleIn = other
#
#     def getHandleOut(self):
#         return self.handleOut
#
#     def setHandleOut(self, other):
#         self.handleOut = other
#
#
# class PathFitter:
#
#     def __init__(self, segments, error=2.5):
#         self.points = []
#         # Copy over points from path and filter out adjacent duplicates.
#         l = len(segments)
#         prev = None
#         for i in range(l):
#             point = segments[i].point.copy()
#             if prev != point:
#                 self.points.append(point)
#                 prev = point
#         self.error = error
#
#     def fit(self):
#         points = self.points
#         length = len(points)
#         self.segments = [Segment(points[0])] if length > 0 else []
#         if length > 1:
#             self.fitCubic(0, length - 1,
#                           # Left Tangent
#                           points[1].subtract(points[0]).normalize(),
#                           # Right Tangent
#                           points[length - 2].subtract(points[length - 1]).normalize())
#         return self.segments
#
#     # Fit a Bezier curve to a (sub)set of digitized points
#     def fitCubic(self, first, last, tan1, tan2):
#         #  Use heuristic if region only has two points in it
#         if last - first == 1:
#             pt1 = self.points[first]
#             pt2 = self.points[last]
#             dist = pt1.getDistance(pt2) / 3
#             self.addCurve([pt1, pt1 + tan1.normalize(dist),
#                            pt2 + tan2.normalize(dist), pt2])
#             return
#         # Parameterize points, and attempt to fit curve
#         uPrime = self.chordLengthParameterize(first, last)
#         maxError = max(self.error, self.error * self.error)
#         # Try 4 iterations
#         for i in range(5):
#             curve = self.generateBezier(first, last, uPrime, tan1, tan2)
#             #  Find max deviation of points to fitted curve
#             maxerr, maxind = self.findMaxError(first, last, curve, uPrime)
#             if maxerr < self.error:
#                 self.addCurve(curve)
#                 return
#             split = maxind
#             # If error not too large, try reparameterization and iteration
#             if maxerr >= maxError:
#                 break
#             self.reparameterize(first, last, uPrime, curve)
#             maxError = maxerr
#         # Fitting failed -- split at max error point and fit recursively
#         V1 = self.points[split - 1].subtract(self.points[split])
#         V2 = self.points[split] - self.points[split + 1]
#         tanCenter = V1.add(V2).divide(2).normalize()
#         self.fitCubic(first, split, tan1, tanCenter)
#         self.fitCubic(split, last, tanCenter.negate(), tan2)
#
#     def addCurve(self, curve):
#         prev = self.segments[len(self.segments) - 1]
#         prev.setHandleOut(curve[1].subtract(curve[0]))
#         self.segments.append(
#             Segment(curve[3], curve[2].subtract(curve[3])))
#
#     # Use least-squares method to find Bezier control points for region.
#     def generateBezier(self, first, last, uPrime, tan1, tan2):
#         epsilon = 1e-11
#         pt1 = self.points[first]
#         pt2 = self.points[last]
#         # Create the C and X matrices
#         C = [[0, 0], [0, 0]]
#         X = [0, 0]
#
#         l = last - first + 1
#
#         for i in range(l):
#             u = uPrime[i]
#             t = 1 - u
#             b = 3 * u * t
#             b0 = t * t * t
#             b1 = b * t
#             b2 = b * u
#             b3 = u * u * u
#             a1 = tan1.normalize(b1)
#             a2 = tan2.normalize(b2)
#             tmp = (self.points[first + i]
#                    - pt1.multiply(b0 + b1)
#                    - pt2.multiply(b2 + b3))
#             C[0][0] += a1.dot(a1)
#             C[0][1] += a1.dot(a2)
#             # C[1][0] += a1.dot(a2)
#             C[1][0] = C[0][1]
#             C[1][1] += a2.dot(a2)
#             X[0] += a1.dot(tmp)
#             X[1] += a2.dot(tmp)
#
#         # Compute the determinants of C and X
#         detC0C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
#         if abs(detC0C1) > epsilon:
#             # Kramer's rule
#             detC0X = C[0][0] * X[1] - C[1][0] * X[0]
#             detXC1 = X[0] * C[1][1] - X[1] * C[0][1]
#             # Derive alpha values
#             alpha1 = detXC1 / detC0C1
#             alpha2 = detC0X / detC0C1
#         else:
#             # Matrix is under-determined, try assuming alpha1 == alpha2
#             c0 = C[0][0] + C[0][1]
#             c1 = C[1][0] + C[1][1]
#             if abs(c0) > epsilon:
#                 alpha1 = alpha2 = X[0] / c0
#             elif abs(c1) > epsilon:
#                 alpha1 = alpha2 = X[1] / c1
#             else:
#                 # Handle below
#                 alpha1 = alpha2 = 0
#
#         # If alpha negative, use the Wu/Barsky heuristic (see text)
#         # (if alpha is 0, you get coincident control points that lead to
#         # divide by zero in any subsequent NewtonRaphsonRootFind() call.
#         segLength = pt2.getDistance(pt1)
#         epsilon *= segLength
#         if alpha1 < epsilon or alpha2 < epsilon:
#             # fall back on standard (probably inaccurate) formula,
#             # and subdivide further if needed.
#             alpha1 = alpha2 = segLength / 3
#
#         # First and last control points of the Bezier curve are
#         # positioned exactly at the first and last data points
#         # Control points 1 and 2 are positioned an alpha distance out
#         # on the tangent vectors, left and right, respectively
#         return [pt1, pt1.add(tan1.normalize(alpha1)),
#                 pt2.add(tan2.normalize(alpha2)), pt2]
#
#     # Given set of points and their parameterization, try to find
#     # a better parameterization.
#     def reparameterize(self, first, last, u, curve):
#         for i in range(first, last + 1):
#             u[i - first] = self.findRoot(curve, self.points[i], u[i - first])
#
#     # Use Newton-Raphson iteration to find better root.
#     def findRoot(self, curve, point, u):
#         # Generate control vertices for Q'
#         curve1 = [
#             curve[i + 1].subtract(curve[i]).multiply(3) for i in range(3)]
#         # Generate control vertices for Q''
#         curve2 = [
#             curve1[i + 1].subtract(curve1[i]).multiply(2) for i in range(2)]
#         # Compute Q(u), Q'(u) and Q''(u)
#         pt = self.evaluate(3, curve, u)
#         pt1 = self.evaluate(2, curve1, u)
#         pt2 = self.evaluate(1, curve2, u)
#         diff = pt - point
#         df = pt1.dot(pt1) + diff.dot(pt2)
#         # Compute f(u) / f'(u)
#         if abs(df) < TOLERANCE:
#             return u
#         # u = u - f(u) / f'(u)
#         return u - diff.dot(pt1) / df
#
#     # Evaluate a bezier curve at a particular parameter value
#     def evaluate(self, degree, curve, t):
#         # Copy array
#         tmp = curve[:]
#         # Triangle computation
#         for i in range(1, degree + 1):
#             for j in range(degree - i + 1):
#                 tmp[j] = tmp[j].multiply(1 - t) + tmp[j + 1].multiply(t)
#         return tmp[0]
#
#     # Assign parameter values to digitized points
#     # using relative distances between points.
#     def chordLengthParameterize(self, first, last):
#         u = {0: 0}
#         print(first, last)
#         for i in range(first + 1, last + 1):
#             u[i - first] = u[i - first - 1] + \
#                 self.points[i].getDistance(self.points[i - 1])
#         m = last - first
#         for i in range(1, m + 1):
#             u[i] /= u[m]
#         return u
#
#     # Find the maximum squared distance of digitized points to fitted curve.
#     def findMaxError(self, first, last, curve, u):
#         index = math.floor((last - first + 1) / 2)
#         maxDist = 0
#         for i in range(first + 1, last):
#             P = self.evaluate(3, curve, u[i - first])
#             v = P.subtract(self.points[i])
#             dist = v.x * v.x + v.y * v.y  # squared
#             if dist >= maxDist:
#                 maxDist = dist
#                 index = i
#         return maxDist, index
#
#
# def fitpath(pointlist, error):
#     return PathFitter(list(map(Segment, map(Point, pointlist))), error).fit()
#
#
# def fitpathsvg(pointlist, error):
#     return pathtosvg(PathFitter(list(map(Segment, map(Point, pointlist))), error).fit())
#
#
# def pathtosvg(path):
#     segs = ['M', str(path[0].point)]
#     last = path[0]
#     for seg in path[1:]:
#         segs.append('C')
#         segs.append(str(last.point + last.handleOut))
#         segs.append(str(seg.point + seg.handleIn))
#         segs.append(str(seg.point))
#         last = seg
#     return ' '.join(segs)
#
#
# if __name__ == '__main__':
#     p = ((88, 151), (90, 151), (98, 151), (105, 151), (112, 151), (121, 151), (141, 151), (153, 150), (165, 150), (203, 150), (224, 151),
#          (268, 154), (282, 155), (331, 156), (340, 156), (353, 156), (358, 156), (361, 156), (362, 156), (365, 156), (366, 156), (372, 156), (373, 156))
#     pf = fitpath(p, error=2.5)
#     print(pf)
#     sp = pathtosvg(pf)
#     print(sp)


# src_box = getRotatedMinimumBoundingBox(src_img)
#
# img_rgb = cv2.line(img_rgb, (src_box[0][0], src_box[0][1]), (src_box[1][0], src_box[1][1]), (0, 0, 255), 1)
# img_rgb = cv2.line(img_rgb, (src_box[1][0], src_box[1][1]), (src_box[2][0], src_box[2][1]), (0, 0, 255), 1)
# img_rgb = cv2.line(img_rgb, (src_box[2][0], src_box[2][1]), (src_box[3][0], src_box[3][1]), (0, 0, 255), 1)
# img_rgb = cv2.line(img_rgb, (src_box[3][0], src_box[3][1]), (src_box[0][0], src_box[0][1]), (0, 0, 255), 1)

# print(src_box)
#
# print(len(src_box))

# import math
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# from utils.Functions import getNumberOfValidPixels
#
# src_path = "../strokes/test.png"
#
# src_img = cv2.imread(src_path, 0)
# img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
#
#
# # threshold
# _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
#
#
# src_img_ = src_img != 255
#
# print(src_img_)
# src_skel = skeletonize(src_img_)
#
# src_skel = (1 - src_skel) * 255
#
# src_skel = np.array(src_skel, dtype=np.uint8)
#
# src_skel_rgb = cv2.cvtColor(src_skel, cv2.COLOR_GRAY2BGR)
#
# # end points of lines
# end_points_list = []
# for y in range(1, src_skel.shape[0]-1):
#     for x in range(1, src_skel.shape[1]-1):
#         if src_skel[y][x] == 0.0:
#             # black points
#             black_num = getNumberOfValidPixels(src_skel, x, y)
#
#             # end points
#             if black_num == 1:
#                 print(black_num)
#                 src_skel_rgb[y][x] = (0, 0, 255)
#                 end_points_list.append((x, y))
#
# print('len of end points: %d' % len(end_points_list))
#
# # cross points
# DIST_THRESHOLD = 20
# cross_points_list = []
# for y in range(1, src_skel.shape[0]-1):
#     for x in range(1, src_skel.shape[1]-1):
#         if src_skel[y][x] == 0.0:
#             # black
#             black_num = getNumberOfValidPixels(src_skel, x, y)
#
#             # normal point = 2 or not normal
#             if black_num > 2:
#                 for (x_, y_) in end_points_list:
#                     dist_ = math.sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_))
#                     if dist_ > DIST_THRESHOLD:
#                         # true cross point
#                         print(black_num)
#                         src_skel_rgb[y][x] = (255, 0, 0)
#                         cross_points_list.append((x, y))
#                         break
#
# for y in range(src_img.shape[0]):
#     for x  in range(src_img.shape[1]):
#         if src_skel[y][x] != 255:
#             img_rgb[y][x] = (255, 0, 0)
#
# print('len of cross points: %d ' % len(cross_points_list))
#
# print(np.min(src_skel), np.max(src_skel))
#
# print(src_skel.shape)
#
# print(src_skel)
# cv2.imshow('src', src_img)
# cv2.imshow('skel', src_skel)
# cv2.imshow('skel rgb', src_skel_rgb)
# cv2.imshow('img rgb', img_rgb)
#
# # cv2.imshow('src', img_rgb)
#
# # cv2.imwrite('skele.png', src_skel)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2

source_path = "../templates/templates/ben/char/ben.png"

source_img = cv2.imread(source_path, 0)
_, source_img = cv2.threshold(source_img, 127, 255, cv2.THRESH_BINARY)

# target image
target_path = "../templates/templates_comparison/ben/char/ben.png"
target_img = cv2.imread(target_path, 0)
_, target_img = cv2.threshold(target_img, 127, 255, cv2.THRESH_BINARY)
print(target_img.shape)
# resize target image
new_tg_h = max(source_img.shape[0], target_img.shape[0])
new_tg_w = max(source_img.shape[1], target_img.shape[1])

target_img = cv2.resize(target_img, (new_tg_h, new_tg_w))

stroke_list = []
stroke_list.append((1477, 1571, 187, 93))
stroke_list.append((1505, 1350, 159, 314))
stroke_list.append((1402, 1466, 262, 198))
stroke_list.append((1541, 1616, 123, 48))
stroke_list.append((1622, 1405, 42, 259))

target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)

for stroke in stroke_list:
    target_img_rgb[stroke[1]: stroke[1]+stroke[3], stroke[0]: stroke[0]+stroke[2]] = (255, 0, 0)
    break



target_img_rgb = cv2.resize(target_img_rgb, (int(target_img_rgb.shape[0]/2), int(target_img_rgb.shape[1]/2)))

cv2.imshow("rgb", target_img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()



