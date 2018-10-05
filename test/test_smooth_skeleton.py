import cv2
import numpy as np
import math

from utils.Functions import getSkeletonOfImage, getCrossPointsOfSkeletonLine, getEndPointsOfSkeletonLine, \
    removeShortBranchesOfSkeleton


path = "test_images/page1_char_3.png"
img = cv2.imread(path, 0)

_, img_bit = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# get skeleton
skeleton = getSkeletonOfImage(img_bit)


cv2.imshow("skeleton smoothed", skeleton)


cv2.waitKey(0)
cv2.destroyAllWindows()