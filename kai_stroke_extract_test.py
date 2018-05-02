# coding: utf-8
import cv2
import numpy as np
from skimage.feature import match_template
from matplotlib import pyplot as plt

from utils.Functions import getContourOfImage, sortPointsOnContourOfImage, getSingleMaxBoundingBoxOfImage


def main():
    img_path = "0001ding.jpg"
    template_path = "0001ding_stroke.jpg"

    img = cv2.imread(img_path, 0)
    temp_img = cv2.imread(template_path, 0)

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    temp_x, temp_y, temp_w, temp_h = getSingleMaxBoundingBoxOfImage(temp_img)

    temp_img = temp_img[temp_x:temp_x+temp_w, temp_y:temp_y+temp_h]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img_ = img.copy()
        method = eval(meth)

        res = cv2.matchTemplate(img_, temp_img, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + temp_w, top_left[1] + temp_h)

        cv2.rectangle(img_rgb, top_left, bottom_right, (0,0,255), 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_rgb)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

    # contour = getContourOfImage(img)
    #
    # # sorted the contour with clockwise driection
    # contour_sorted = sortPointsOnContourOfImage(contour)
    # print(contour_sorted)


    # cv2.imshow("src", img)
    # # cv2.imshow("contour", contour)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()