import cv2
import numpy as np
from collections import namedtuple
from skimage.morphology import skeletonize
from utils.Functions import getSkeletonOfImage, getEndPointsOfSkeletonLine, getCrossAreaPointsOfSkeletionLine, \
    removeBranchOfSkeletonLine, getBoundingBoxes, getCrossPointsOfSkeletonLine, splitConnectedComponents


def main():
    # Point
    Point = namedtuple("Point", ["x", "y"])

    # target image
    target_path = "../templates/ben.png"
    target_img = cv2.imread(target_path, 0)
    _, target_img = cv2.threshold(target_img, 127, 255, cv2.THRESH_BINARY)
    print(target_img.shape)

    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)

    # connected components with labeling algorithm
    partial_parts = splitConnectedComponents(target_img)
    print("number of parts: %d" % len(partial_parts))
    # part 1
    part_1 = partial_parts[0]

    # skeleton line
    part_1_ = part_1 != 255

    part_skel = skeletonize(part_1_)
    part_skel = (1 - part_skel) * 255

    part_skel = np.array(part_skel, dtype=np.uint8)

    for y in range(part_skel.shape[0]):
        for x in range(part_skel.shape[1]):
            if part_skel[y][x] == 0.0:
                target_img_rgb[y][x] = (0, 255, 0)

    # remove extra branch
    end_points = getEndPointsOfSkeletonLine(part_skel)
    cross_points = getCrossPointsOfSkeletonLine(part_skel)

    part_skel = removeBranchOfSkeletonLine(part_skel, end_points, cross_points)

    part_1_rgb_no_branch = cv2.cvtColor(part_1, cv2.COLOR_GRAY2RGB)

    for y in range(part_skel.shape[0]):
        for x in range(part_skel.shape[1]):
            if part_skel[y][x] == 0.0:
                part_1_rgb_no_branch[y][x] = (0, 255, 0)

    # new end points and cross points without extra branches
    end_points = getEndPointsOfSkeletonLine(part_skel)
    print("number of end points: %d" % len(end_points))
    cross_points = getCrossPointsOfSkeletonLine(part_skel)
    print("number of cross points: %d" % len(cross_points))

    # add end points to image with blue color
    for (x, y) in end_points:
        part_1_rgb_no_branch[y][x] = (255, 0, 0)

    # add cross points to image with red color
    for (x, y) in cross_points:
        part_1_rgb_no_branch[y][x] = (0, 0, 255)

    # Contour  of character
    part_1_edges = cv2.Canny(part_1, 100, 200)
    part_1_edges = 255 - part_1_edges

    # Order the points on contours of character
    print(part_1_edges.shape)
    begin_point = None

    # find the begin point
    for y in range(part_1_edges.shape[0]):
        for x in range(part_1_edges.shape[1]):
            if part_1_edges[y][x] == 0.0:
                # first black point
                begin_point = (x, y)
                break
        if begin_point:
            break
    print("begin point: (%d, %d)" % (begin_point[0], begin_point[1]))

    edge_order_lables = np.zeros_like(part_1_edges)
    edge_order_lables[begin_point[1]][begin_point[0]] = 1.

    curr_point = begin_point

    # find the second point
    if part_1_edges[y][x+1] == 0.0:
        print("Second point is 4 position")
        curr_point = (x+1, y)
    elif part_1_edges[y+1][x+1] == 0.0:
        print("Second point is 5 position")
        curr_point = (x+1, y+1)
    print(curr_point)
    edge_order_lables[curr_point[1]][curr_point[0]] = 1.

    edge_points = []
    edge_points.append(begin_point)
    edge_points.append(curr_point)

    next_point = curr_point
    edge_id = 0
    while True:

        x = curr_point[0]; y = curr_point[1]
        # 2,4,6,8 position firstly and then 3,5,7,9 position
        # point in 2 position
        if part_1_edges[y-1][x] == 0.0 and edge_order_lables[y-1][x] == 0.0:
            print("%d po" % 2)
            next_point = (x, y-1)
            edge_order_lables[y-1][x] = 1.

        # point in 4 position
        elif part_1_edges[y][x + 1] == 0.0 and edge_order_lables[y][x + 1] == 0.0:
            print("%d po" % 4)
            next_point = (x + 1, y)
            edge_order_lables[y][x + 1] = 1.

        # point in 6 position
        elif part_1_edges[y + 1][x] == 0.0 and edge_order_lables[y + 1][x] == 0.0:
            print("%d po" % 6)
            next_point = (x, y + 1)
            edge_order_lables[y + 1][x] = 1.

        # point in 8 position
        elif part_1_edges[y][x - 1] == 0.0 and edge_order_lables[y][x - 1] == 0.0:
            print("%d po" % 8)
            next_point = (x - 1, y)
            edge_order_lables[y][x - 1] = 1.

        # point in 3 position
        elif part_1_edges[y-1][x+1] == 0.0 and edge_order_lables[y-1][x+1] == 0.0:
            print("%d po" % 3)
            next_point = (x+1, y-1)
            edge_order_lables[y-1][x+1] = 1.

        # point in 5 position
        elif part_1_edges[y+1][x+1] == 0.0 and edge_order_lables[y+1][x+1] == 0.0:
            print("%d po" % 5)
            next_point = (x+1, y+1)
            edge_order_lables[y+1][x+1] = 1.

        # point in 7 position
        elif part_1_edges[y+1][x-1] == 0.0 and edge_order_lables[y+1][x-1] == 0.0:
            print("%d po" % 7)
            next_point = (x-1, y+1)
            edge_order_lables[y+1][x-1] = 1.

        # point in 9 position
        elif part_1_edges[y-1][x-1] == 0.0 and edge_order_lables[y-1][x-1] == 0.0:
            print("%d po" % 9)
            next_point = (x-1, y-1)
            edge_order_lables[y-1][x-1] = 1.

        if next_point == curr_point:
            print(next_point)
            print(curr_point)
            edge_points.append(curr_point)
            break
        else:
            edge_points.append(curr_point)
            edge_id += 1
            print("edge id: %d" % edge_id)
            curr_point = next_point

    print("edge points len: %d" % len(edge_points))
    for pt in edge_points:
        print(pt)
        part_1_rgb_no_branch[pt[1]][pt[0]] = (0, 0, 255)






    # # houngh lines
    # rho_resolution = 1
    # theta_resolution = np.pi / 180
    # threshold = 155
    # hough_lines = cv2.HoughLines(part_1_edges, rho_resolution, theta_resolution, threshold)
    #
    # print("number of hough lines: %d " % len(hough_lines))
    #
    # hough_lines_img = np.zeros_like(part_1_edges)
    # draw_lines(hough_lines_img, hough_lines)
    # original_image_with_hough_lines = weighted_img(hough_lines_img, part_1_edges)
    #
    # cv2.imshow("hough line image", hough_lines_img)
    # cv2.imshow("original hough", original_image_with_hough_lines)


    # # Corner detection
    # corners = cv2.goodFeaturesToTrack(part_1_edges, 100, 0.01, 10)
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(part_1_rgb_no_branch, (x, y), 3, 255, -1)

    cv2.imshow("img", target_img)
    # cv2.imshow("skel", part_skel)
    # cv2.imshow("img_rgb", target_img_rgb)
    cv2.imshow("img_rgb_no_branch", part_1_rgb_no_branch)
    cv2.imshow("edges", part_1_edges)

    cv2.imwrite("../templates/ben_skeleton.png", target_img_rgb)
    cv2.imwrite("../templates/ben_skeleton_no_branch.png", part_1_rgb_no_branch)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):
    for line in houghLines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, alpha=0.8, beta=1., gama=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gama)

if __name__ == '__main__':
    main()