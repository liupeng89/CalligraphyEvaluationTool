import cv2
import numpy as np

from utils.Functions import getConvexHullOfImage


def main():
    src_path = "src_resize.png"
    tag_path = "tag_resize.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_l = getConvexHullOfImage(src_img)
    tag_l = getConvexHullOfImage(tag_img)

    print("src L len: %d" % len(src_l))
    print("tag L len: %d" % len(tag_l))

    src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

    # Source convex hull
    for idx in range(len(src_l)):
        if idx+1 == len(src_l):
            src_img_rgb = cv2.line(src_img_rgb, (src_l[idx][1], src_l[idx][0]), (src_l[0][1], src_l[0][0]), (0, 0, 255), 3)
        else:
            src_img_rgb = cv2.line(src_img_rgb, (src_l[idx][1], src_l[idx][0]), (src_l[idx+1][1], src_l[idx+1][0]), (0, 0, 255), 3)

    # Target convex hull
    for idx in range(len(tag_l)):
        if idx+1 == len(tag_l):
            tag_img_rgb = cv2.line(tag_img_rgb, (tag_l[idx][1], tag_l[idx][0]), (tag_l[0][1], tag_l[0][0]), (0, 0, 255), 3)
        else:
            tag_img_rgb = cv2.line(tag_img_rgb, (tag_l[idx][1], tag_l[idx][0]), (tag_l[idx+1][1], tag_l[idx+1][0]), (0, 0, 255), 3)

    cv2.imshow("src", src_img_rgb)
    cv2.imshow("tag", tag_img_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()