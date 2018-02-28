import cv2
import numpy as np
import math
from utils.Functions import resizeImages, getNumberOfValidPixels
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


def getEndPointsOfSkeletonLine(image):
    """
        Obtain the end points of skeleton line, suppose the image is the skeleton image(white background and black
        skeleton line).
    :param image:
    :return: the end points of skeleton line
    """
    end_points = []
    if image is None:
        return end_points

    # find the end points which number == 1
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y][x] == 0.0:
                # black points
                black_num = getNumberOfValidPixels(image, x, y)

                # end points
                if black_num == 1:
                    end_points.append((x, y))
    return end_points


def getCrossPointsOfSkeletonLine(image):
    """
        Get the cross points of skeleton line to find the extra branch.
    :param image:
    :return: coordinate of cross points
    """
    cross_points = []
    cross_points_no_extra = []
    if image is None:
        return cross_points
    # find cross points
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y][x] == 0.0:
                # black points
                black_num = getNumberOfValidPixels(image, x, y)

                # cross points
                if black_num >= 3:
                    cross_points.append((x, y))
    print("cross points len : %d" % len(cross_points))

    # remove the extra cross points and maintain the single cross point of several close points
    for (x, y) in cross_points:
        black_num = 0
        # P2
        if image[y-1][x] == 0.0 and (x, y-1) in cross_points:
            black_num += 1
        # P4
        if image[y][x+1] == 0.0 and (x+1, y) in cross_points:
            black_num += 1
        # P6
        if image[y+1][x] == 0.0 and (x, y+1) in cross_points:
            black_num += 1
        # P8
        if image[y][x-1] == 0.0 and (x-1, y) in cross_points:
            black_num += 1

        if black_num == 2 or black_num == 3 or black_num == 4:
            cross_points_no_extra.append((x, y))

    return cross_points_no_extra


def removeBranchOfSkeletonLine(image, end_points, cross_points):
    """
        Remove brches of skeleton line.
    :param image:
    :param end_points:
    :param cross_points:
    :return:
    """
    if image is None:
        return None
    # image = image.copy()
    # remove branches of skeleton line
    for (c_x, c_y) in cross_points:
        for (e_x, e_y) in end_points:
            dist = math.sqrt((c_x-e_x) * (c_x-e_x) + (c_y-e_y) * (c_y-e_y))
            if dist < DIST_THRESHOLD:
                print("%d %d %d %d " % (e_x, e_y, c_x, c_y))
                branch_points = getPointsOfExtraBranchOfSkeletonLine(image, e_x, e_y, c_x, c_y)
                print("branch length: %d" % len(branch_points))

                # remove branch points
                for (bx, by) in branch_points:
                    image[by][bx] = 255

    return image


def getPointsOfExtraBranchOfSkeletonLine(image, start_x, start_y, end_x, end_y):
    """
        Obtain all points of extra branch of skeleton line: end point -> cross point
    :param image:
    :param start_x:
    :param start_y:
    :param end_x:
    :param end_y:
    :return: all points of extra branch
    """
    extra_branch_points = []
    start_point = (start_x, start_y)
    next_point = start_point

    while(True):
        print(start_point)
        print((end_x, end_y))
        print("----")
        # P2
        if image[start_point[1]-1][start_point[0]] == 0.0 and (start_point[0], start_point[1]-1) not in \
                extra_branch_points:
            next_point = (start_point[0], start_point[1]-1)
            print(next_point)
        # P3
        if image[start_point[1]-1][start_point[0]+1] == 0.0 and (start_point[0]+1, start_point[1]-1) not in \
                extra_branch_points:
            next_point = (start_point[0]+1, start_point[1]-1)
            print(next_point)
        # P4
        if image[start_point[1]][start_point[0]+1] == 0.0 and (start_point[0]+1, start_point[1]) not in \
                extra_branch_points:
            next_point = (start_point[0]+1, start_point[1])
            print(next_point)
        # P5
        if image[start_point[1]+1][start_point[0]+1] == 0.0 and (start_point[0]+1, start_point[1]+1) not in \
                extra_branch_points:
            next_point = (start_point[0]+1, start_point[1]+1)
            print(next_point)
        # P6
        if image[start_point[1]+1][start_point[0]] == 0.0 and (start_point[0], start_point[1]+1) not in \
                extra_branch_points:
            next_point = (start_point[0], start_point[1]+1)
            print(next_point)
        # P7
        if image[start_point[1]+1][start_point[0]-1] == 0.0 and (start_point[0]-1, start_point[1]+1) not in \
                extra_branch_points:
            next_point = (start_point[0]-1, start_point[1]+1)
            print(next_point)
        # P8
        if image[start_point[1]][start_point[0]-1] == 0.0 and (start_point[0]-1, start_point[1]) not in \
                extra_branch_points:
            next_point = (start_point[0]-1, start_point[1])
            print(next_point)
        # P9
        if image[start_point[1]-1][start_point[0]-1] == 0.0 and (start_point[0]-1, start_point[1]-1) not in \
                extra_branch_points:
            next_point = (start_point[0]-1, start_point[1]-1)
            print(next_point)

        extra_branch_points.append(start_point)

        if next_point[0] == end_x and next_point[1] == end_y:
            # next point is the cross point
            break
        else:
            start_point = next_point

    return extra_branch_points


if __name__ == '__main__':
    main()