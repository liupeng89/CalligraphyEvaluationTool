import cv2
import numpy as np
from utils.Functions import rotate, resizeImages, calculateBoundingBox, rotate_character


def main():
    src_path = "../chars/src_dan_svg_simple.png"
    tag_path = "../chars/tag_dan_svg_simple.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    # rgt to grayscale

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_img, tag_img = resizeImages(src_img, tag_img)

    dst = rotate_character(src_img, 90)

    cv2.imshow("src img", src_img)
    cv2.imshow("rotate img", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()