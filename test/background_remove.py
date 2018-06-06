import numpy as np
import cv2

path = "seg_test.jpg"

img = cv2.imread(path)

fgbg = cv2.createBackgroundSubtractorMOG2()

fgmask = fgbg.apply(img)

cv2.imshow("mask", fgmask)

cv2.waitKey(0)
cv2.destroyAllWindows()