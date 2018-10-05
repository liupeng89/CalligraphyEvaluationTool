import cv2
import numpy as np
import math

from utils.Functions import getContourImage, getSingleMaxBoundingBoxOfImage, shiftImageWithMaxCoverageArea

char_path = "../test_images/src_comp_0.png"

stroke_1_path = "../test_images/temp_stroke_0.png"
stroke_2_path = "../test_images/temp_stroke_2.png"


char_img = cv2.imread(char_path, 0)
stroke_1_img = cv2.imread(stroke_1_path, 0)
stroke_2_img = cv2.imread(stroke_2_path, 0)


_, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
_, stroke_1_img = cv2.threshold(stroke_1_img, 127, 255, cv2.THRESH_BINARY)
_, stroke_2_img = cv2.threshold(stroke_2_img, 127, 255, cv2.THRESH_BINARY)

# shift with max coverage area
stroke_1_img = shiftImageWithMaxCoverageArea(char_img, stroke_1_img)




# get contours
char_cont = getContourImage(char_img)
stroke_1_cont = getContourImage(stroke_1_img)
stroke_2_cont = getContourImage(stroke_2_img)

# get all black points in contours
char_cont_pixels = []
stroke_1_cont_pixels = []
stroke_2_cont_pixels = []

for i in range(char_cont.shape[0]):
    for j in range(char_cont.shape[1]):
        if char_cont[i][j] == 0.:
            char_cont_pixels.append([i, j])

for i in range(stroke_1_cont.shape[0]):
    for j in range(stroke_1_cont.shape[1]):
        if stroke_1_cont[i][j] == 0.:
            stroke_1_cont_pixels.append([i, j])

for i in range(stroke_2_cont.shape[0]):
    for j in range(stroke_2_cont.shape[1]):
        if stroke_2_cont[i][j] == 0.:
            stroke_2_cont_pixels.append([i, j])
print("char cont num:", len(char_cont_pixels), "stroke1 num: ", len(stroke_1_cont_pixels), "stroke2num:", len(stroke_2_cont_pixels))

stroke_1_nearest_points = []

for pt in stroke_1_cont_pixels:
    min_dist = 10000000000.
    near_pt = None
    for cpt in char_cont_pixels:
        dist_ = math.sqrt((pt[0]-cpt[0])**2 + (pt[1]-cpt[1])**2)
        if dist_ < min_dist:
            min_dist = dist_
            near_pt = cpt

    if near_pt:
        stroke_1_nearest_points.append(near_pt)
print("nearest strokes 1 points num: ", len(stroke_1_nearest_points))

stroke_2_nearest_points = []
for pt in stroke_2_cont_pixels:
    min_dist = 10000000000.
    near_pt = None
    for cpt in char_cont_pixels:
        dist_ = math.sqrt((pt[0]-cpt[0])**2 + (pt[1]-cpt[1])**2)
        if dist_ < min_dist:
            min_dist = dist_
            near_pt = cpt
    if near_pt:
        stroke_2_nearest_points.append(near_pt)
print("nearest strokes 2 points num: ", len(stroke_2_nearest_points))

char_cont_rgb = cv2.cvtColor(char_cont.copy(), cv2.COLOR_GRAY2BGR)
for pt in stroke_1_nearest_points:
    char_cont_rgb[pt[0]][pt[1]] = (0, 0, 255)

for pt in stroke_2_nearest_points:
    char_cont_rgb[pt[0]][pt[1]] = (255, 0, 0)




cv2.imshow("char cont rbg", char_cont_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()



