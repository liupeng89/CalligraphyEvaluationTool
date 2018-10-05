import cv2
import numpy as np

from utils.Functions import shiftImageWithMaxCoverageArea, shiftImageWithMaxCR


char_path = "../test_images/src_comp_0.png"

stroke_path = "../test_images/temp_stroke_0.png"

char_img = cv2.imread(char_path, 0)
stroke_img = cv2.imread(stroke_path, 0)

_, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
_, stroke_img = cv2.threshold(stroke_img, 127, 255, cv2.THRESH_BINARY)


new_stroke_img = shiftImageWithMaxCR(char_img, stroke_img)


cv2.imshow("stroke img", stroke_img)
cv2.imshow("new stroke img", new_stroke_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

