import cv2
import numpy as np
from utils.Functions import resizeImages, coverTwoImages, shiftImageWithMaxCR, calculateCR, addIntersectedFig, addSquaredFig


def main():
    src_path = "../characters/src_dan_processed.png"
    tag_path = "../characters/tag_dan_processed.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    ret, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # resize
    src_img, tag_img = resizeImages(src_img, tag_img)

    # Threshold
    ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    ret, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # cv2.imwrite("src_resize.png", src_img)
    # cv2.imwrite("tag_resize.png", tag_img)

    # Cover Images

    coverage_img = coverTwoImages(src_img, tag_img)
    cr = calculateCR(src_img, tag_img)
    print("No shifting cr: %f" % cr)

    # coverage_img = addIntersectedFig(coverage_img)
    coverage_img = addSquaredFig(coverage_img)

    # Shift images with max CR
    # new_tag_img = shiftImageWithMaxCR(src_img, tag_img)

    # Cover images
    # coverage_img1 = coverTwoImages(src_img, new_tag_img)

    cv2.imshow("coverage img", coverage_img)
    # cv2.imshow("new coverage img", coverage_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()