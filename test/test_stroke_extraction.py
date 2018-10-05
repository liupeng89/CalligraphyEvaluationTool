import cv2
import numpy as np
from algorithms.RDP import rdp
from matplotlib import pyplot as plt

from utils.Functions import getContourOfImage, getSkeletonOfImage, createBlankGrayscaleImage, getConnectedComponents,\
    sortPointsOnContourOfImage, removeBreakPointsOfContour

path = "../test_images/page1_char_3.png"

img = cv2.imread(path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_contour = getContourOfImage(img)
#
# dft1 = cv2.dft(np.float32(img_contour), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift1 = np.fft.fftshift(dft1)

dft_basic = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_basic = np.fft.fftshift(dft_basic)
#
# dft_basic += dft1

dft_ift  = np.fft.ifftshift(dft_basic)
img_reverse = cv2.idft(dft_ift)
img_reverse = cv2.magnitude(img_reverse[:,:,0],img_reverse[:,:,1])

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_basic[:,:,0],dft_basic[:,:,1]))

plt.subplot(121),plt.imshow(img_reverse, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# img_contour = getContourOfImage(img)
# img_contour = removeBreakPointsOfContour(img_contour)
#
# img_contour_rgb = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2RGB)
#
# img_contour_simp = createBlankGrayscaleImage(img_contour)
#
# img_skeleton = getSkeletonOfImage(img)
#
# # simplify the contour
# contours = getConnectedComponents(img_contour, connectivity=8)
#
# contours_points = []
# for cont in contours:
#     cont_sorted = sortPointsOnContourOfImage(cont)
#     cont_simp = rdp(cont_sorted, 2)
#     for i in range(len(cont_simp)):
#         start_pt = cont_simp[i]
#         if i == len(cont_simp) - 1:
#             end_pt = cont_simp[0]
#         else:
#             end_pt = cont_simp[i + 1]
#         cv2.line(img_contour_simp, start_pt, end_pt, 0, 1)
#
#
# cv2.imshow("img_contour_simp", img_contour_simp)
#
#
# # get corner points on contour
#
# corner_region_img = np.float32(img.copy())
# dst = cv2.cornerHarris(corner_region_img, blockSize=5, ksize=5, k=0.04)
# dst = cv2.dilate(dst, None)
#
# # get all points in corners area
# corners_area_points = []
# for y in range(dst.shape[0]):
#     for x in range(dst.shape[1]):
#         if dst[y][x] > 0.2 * dst.max():
#             corners_area_points.append((x, y))
#
# # get all center points of corner area
# corners_img = createBlankGrayscaleImage(img_contour)
# for pt in corners_area_points:
#     corners_img[pt[1]][pt[0]] = 0.0
#     img_contour_rgb[pt[1]][pt[0]] = (0, 255, 0)
#
#
# cv2.imshow("img", img)
# cv2.imshow("contour", img_contour)
# cv2.imshow("skeleton", img_skeleton)
# cv2.imshow("corners_img", corners_img)
# cv2.imshow("img_contour_rgb", img_contour_rgb)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()