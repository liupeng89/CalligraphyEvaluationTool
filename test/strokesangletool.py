import numpy as np
import cv2
from utils.Functions import addIntersectedFig


def main():
    src_path = "../strokes/src_strokes2.png"
    tag_path = "../strokes/tag_strokes2.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

    _, src_contours, src_he = cv2.findContours(src_img, 1, 2)
    _, tag_contours, tag_he = cv2.findContours(tag_img, 1, 2)

    src_cnt = src_contours[0]
    tag_cnt = tag_contours[0]

    rows, cols = src_img.shape
    [vx, vy, x, y] = cv2.fitLine(src_cnt, cv2.DIST_L2, 0, 0.01, 0.01)

    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(src_img_rgb, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    cv2.line(src_img_rgb, (cols - 1, lefty), (0, lefty), (0, 0, 255), 1)

    rows, cols = tag_img.shape
    [vx, vy, x, y] = cv2.fitLine(tag_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(tag_img_rgb, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    cv2.line(tag_img_rgb, (cols - 1, lefty), (0, lefty), (0, 0, 255), 1)

    src_img_rgb_ = addIntersectedFig(src_img_rgb)
    tag_img_rgb_ = addIntersectedFig(tag_img_rgb)

    cv2.imshow("src img", src_img_rgb_)
    cv2.imshow("tag img", tag_img_rgb_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()