import cv2
import numpy as np
import math

from utils.Functions import getSkeletonOfImage, getCrossPointsOfSkeletonLine, getEndPointsOfSkeletonLine, \
    removeShortBranchesOfSkeleton




path = "test_images/1133壬.jpg"

img = cv2.imread(path, 0)

_, img_bit = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# get skeleton
skeleton = getSkeletonOfImage(img_bit)

end_points = getEndPointsOfSkeletonLine(skeleton)
cross_points = getCrossPointsOfSkeletonLine(skeleton)

print("end points num: %d" % len(end_points))
print("cross points num: %d" % len(cross_points))

cv2.imshow("skeleton original", skeleton)

# length threshold
skeleton = removeShortBranchesOfSkeleton(skeleton, length_threshold=30)


cv2.imshow("skeleton smoothed", skeleton)


cv2.waitKey(0)
cv2.destroyAllWindows()