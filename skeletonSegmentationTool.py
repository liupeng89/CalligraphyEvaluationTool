import cv2
import numpy as np
from utils.Functions import resizeImages, getEndPointsOfSkeletonLine, getCrossPointsOfSkeletonLine, removeBranchOfSkeletonLine
from skimage.morphology import skeletonize

DIST_THRESHOLD = 10


def main():
    # src_path = "../strokes/src_strokes4.png"
    src_path = "../chars/src_dan_svg_simple_resized.png"
    tag_path = "../strokes/tag_strokes4.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    ret, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # resize
    src_img, tag_img = resizeImages(src_img, tag_img)

    ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    ret, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # obtain the skeleton of strokes
    src_img_ = src_img != 255
    tag_img_ = tag_img != 255

    src_skel = skeletonize(src_img_)
    tag_skel = skeletonize(tag_img_)

    src_skel = (1 - src_skel) * 255
    tag_skel = (1 - tag_skel) * 255

    src_skel = np.array(src_skel, dtype=np.uint8)
    tag_skel = np.array(tag_skel, dtype=np.uint8)

    src_skel_rgb = cv2.cvtColor(src_skel, cv2.COLOR_GRAY2BGR)
    tag_skel_rgb = cv2.cvtColor(tag_skel, cv2.COLOR_GRAY2BGR)

    src_end_points = getEndPointsOfSkeletonLine(src_skel)
    tag_end_points = getEndPointsOfSkeletonLine(tag_skel)

    for (x, y) in src_end_points:
        src_skel_rgb[y][x] = (0, 0, 255)
    for (x, y) in tag_end_points:
        tag_skel_rgb[y][x] = (0, 0, 255)

    print("src end points len: %d" % len(src_end_points))
    print("tag end points len: %d" % len(tag_end_points))

    if len(src_end_points) > 2:
        print("src skeleton line has branch")
    if len(tag_end_points) > 2:
        print("tag skeleton line has branch")
    src_cross_points = getCrossPointsOfSkeletonLine(src_skel)
    tag_cross_points = getCrossPointsOfSkeletonLine(tag_skel)

    for (x, y) in src_cross_points:
        src_skel_rgb[y][x] = (255, 0, 0)
    for (x, y) in tag_cross_points:
        tag_skel_rgb[y][x] = (255, 0, 0)

    print("src cross len: %d" % len(src_cross_points))
    print("tag cross len: %d" % len(tag_cross_points))

    if len(src_cross_points) > 0:
        # exist branches
        src_skel = removeBranchOfSkeletonLine(src_skel, src_end_points, src_cross_points)

    if len(tag_cross_points) > 0:
        # exist branches
        tag_skel = removeBranchOfSkeletonLine(tag_skel, tag_end_points, tag_cross_points)

    # src_skel_no_branch = removeBranchOfSkeletonLine(src_skel, src_end_points, src_cross_points)
    # tag_skel_no_btranch = removeBranchOfSkeletonLine(tag_skel, tag_end_points, tag_cross_points)

    cv2.imshow("coverage img", src_img)
    cv2.imshow("new coverage img", tag_img)

    cv2.imshow("src rgb", src_skel_rgb)
    cv2.imshow("tag rgb", tag_skel_rgb)

    cv2.imshow("src skeleton img", src_skel)
    cv2.imshow("tag skeleton img", tag_skel)

    # cv2.imshow("src skeleton img no branch", src_skel_no_branch)
    # cv2.imshow("tag skeleton img no branch", tag_skel_no_branch)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()