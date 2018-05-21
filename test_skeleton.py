# coding: utf-8
import cv2
import numpy as np

from utils.Functions import getSkeletonOfImage, getEndPointsOfSkeletonLine, getCrossPointsOfSkeletonLine, \
    removeExtraBranchesOfSkeleton

# 1133壬 2252支 0631叟 0633口 0242俄 0195佛 0860善 0059乘 0098亩
path = "2252支.jpg"

img = cv2.imread(path, 0)
_, img_bit = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

skeleton = getSkeletonOfImage(img_bit)
skeleton = removeExtraBranchesOfSkeleton(skeleton)

skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)

end_points = getEndPointsOfSkeletonLine(skeleton)

cross_points = getCrossPointsOfSkeletonLine(skeleton)

# end points
for pt in end_points:
    skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)

for pt in cross_points:
    skeleton_rgb[pt[1]][pt[0]] = (0, 255, 0)


cv2.imshow("skeleton rgb", skeleton_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()


