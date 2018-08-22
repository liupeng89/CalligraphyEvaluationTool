import sys
import math
import cv2
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from calligraphySmoothingTool.smoothmanuallymainwindow import Ui_MainWindow

from utils.Functions import getContourOfImage, removeBreakPointsOfContour, \
                            sortPointsOnContourOfImage, fitCurve, draw_cubic_bezier
from utils.contours_smoothed_algorithm import autoSmoothContoursOfCharacter

path = "../test_images/src_resize.png"

img = cv2.imread(path, 0)

img_smoothed = autoSmoothContoursOfCharacter(img)



cv2.imshow("img", img)
cv2.imshow("img_smoothed", img_smoothed)

cv2.waitKey(0)
cv2.destroyAllWindows()

