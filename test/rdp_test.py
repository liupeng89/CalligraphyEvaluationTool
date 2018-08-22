from algorithms.RDP import rdp
import cv2
import numpy as np

from utils.Functions import getContourOfImage, sortPointsOnContourOfImage, createBlankRGBImage, fitCurve, \
    draw_cubic_bezier, removeBreakPointsOfContour, splitConnectedComponents

from utils.contours_smoothed_algorithm import autoSmoothContoursOfCharacter

path = "../test_images/src_resize.png"

img = cv2.imread(path, 0)

img_smoothed = autoSmoothContoursOfCharacter(img, eplison=30, max_error=20)

cv2.imshow("img", img)
cv2.imshow("img_smoothed", img_smoothed)

cv2.waitKey(0)
cv2.destroyAllWindows()

# contour_img = getContourOfImage(img)
#
# contours = splitConnectedComponents(contour_img)
# print("contours num:", len(contours))




# contour = removeBreakPointsOfContour(contours[0])
#
# contour_sort = sortPointsOnContourOfImage(contour)
# print(contour_sort)
# print(len(contour_sort))
#
# contour_simp = rdp(contour_sort, 10)
# print(contour_simp)
# print(len(contour_simp))
#
# contour_rbg = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
# contour_rbg_ = createBlankRGBImage(contour)
# contour_rbg__ = contour_rbg_.copy()
#
# for pt in contour_simp:
#     contour_rbg[pt[1]][pt[0]] = (0, 0, 255)
#     cv2.circle(contour_rbg_, pt, 2, (255, 0, 0))
#
# for i in range(len(contour_simp)):
#     if i == len(contour_simp) - 1:
#         start_pt = contour_simp[i]
#         end_pt = contour_simp[0]
#     else:
#         start_pt = contour_simp[i]
#         end_pt = contour_simp[i + 1]
#     cv2.line(contour_rbg, start_pt, end_pt, (0, 0, 255), 1)
#     cv2.line(contour_rbg_, start_pt, end_pt, (0, 0, 255), 1)
#
# cv2.circle(contour_rbg, contour_sort[0], 3, (255, 0, 5), 3)
# cv2.circle(contour_rbg, contour_sort[-1], 3, (255, 0, 5), 3)
#
# # segment contour into sub-contours based on the simplified points
# sub_contours = []
# for i in range(len(contour_simp)-1):
#     start_pt = contour_simp[i]
#     end_pt = contour_simp[i+1]
#
#     start_index = contour_sort.index(start_pt)
#     end_index = contour_sort.index(end_pt)
#
#     sub_contours.append(contour_sort[start_index: end_index+1])
#
# print("sub-contours len:", len(sub_contours))
#
# # two closed sub-contours should be merged together.
# # sub_contours_merged = []
# # for i in range(len(sub_contours)):
# #     sub_a = sub_contours[i]
# #     if i == len(sub_contours) - 1:
# #         sub_b = sub_contours[0]
# #     else:
# #         sub_b = sub_contours[i + 1]
# #
# #     merge = sub_a + sub_b
# #
# #     sub_contours_merged.append(merge)
#
# sub_contours_smoothed = []
# max_error = 40.
#
# for id in range(len(sub_contours)):
#     # single sub-contour
#     sub_contour = np.array(sub_contours[id])
#
#     if len(sub_contour) < 2:
#         continue
#     beziers = fitCurve(sub_contour, maxError=max_error)
#     sub_contour_smoothed = []
#
#     for bez in beziers:
#         bezier_points = draw_cubic_bezier(bez[0], bez[1], bez[2], bez[3])
#         sub_contour_smoothed += bezier_points
#
#     sub_contours_smoothed.append(sub_contour_smoothed)
#
# for sub in sub_contours_smoothed:
#     for pt in sub:
#         contour_rbg__[pt[1]][pt[0]] = (255, 0, 0)
#
# cv2.imshow("img", img)
# cv2.imshow("contour", contour)
# cv2.imshow("contour_rgb", contour_rbg)
# cv2.imshow("contour_rgb_", contour_rbg_)
# cv2.imshow("contour_rgb__", contour_rbg__)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


