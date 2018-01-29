import cv2
import numpy as np

from utils.Functions import getConvexHullOfImage, getPolygonArea, getValidPixelsArea, addIntersectedFig


def main():
    src_path = "../chars/src_dan_svg_simple_resized.png"
    tag_path = "../chars/tag_dan_svg_simple_resized.png"

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
            src_img_rgb = cv2.line(src_img_rgb, (src_l[idx][0], src_l[idx][1]), (src_l[0][0], src_l[0][1]), (0, 0, 255), 2)
        else:
            src_img_rgb = cv2.line(src_img_rgb, (src_l[idx][0], src_l[idx][1]), (src_l[idx+1][0], src_l[idx+1][1]), (0, 0, 255), 2)
    src_convexhull_area = getPolygonArea(src_l)
    src_valid_area = getValidPixelsArea(src_img)
    src_area_ratio = src_convexhull_area / (src_img.shape[0] * src_img.shape[1]) * 100
    src_valid_ratio = src_valid_area / src_convexhull_area * 100
    print("src area: %0.3f (%0.3f)" % (src_convexhull_area, src_area_ratio))
    print("src valid area: %0.3f (%0.3f)" % (src_valid_area, src_valid_ratio))


    # Target convex hull
    for idx in range(len(tag_l)):
        if idx+1 == len(tag_l):
            tag_img_rgb = cv2.line(tag_img_rgb, (tag_l[idx][0], tag_l[idx][1]), (tag_l[0][0], tag_l[0][1]), (0, 0, 255), 2)
        else:
            tag_img_rgb = cv2.line(tag_img_rgb, (tag_l[idx][0], tag_l[idx][1]), (tag_l[idx+1][0], tag_l[idx+1][1]), (0, 0, 255), 2)
    tag_convexhull_area = getPolygonArea(tag_l)
    tag_valid_area = getValidPixelsArea(tag_img)
    tag_area_ratio = tag_convexhull_area / (tag_img.shape[0] * tag_img.shape[1]) * 100
    tag_valid_ratio = tag_valid_area / tag_convexhull_area * 100
    print("tag area: %0.3f (%0.3f)" % (tag_convexhull_area, tag_area_ratio))
    print("tag valid area: %0.3f (%0.3f)" % (tag_valid_area, tag_valid_ratio))

    #
    src_img_rgb_ = addIntersectedFig(src_img_rgb)
    tag_img_rgb_ = addIntersectedFig(tag_img_rgb)

    cv2.imshow("src", src_img_rgb_)
    cv2.imshow("tag", tag_img_rgb_)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()