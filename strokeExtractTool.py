import cv2
import numpy as np
from skimage.morphology import skeletonize
from utils.Functions import getSkeletonOfImage, getEndPointsOfSkeletonLine, getCrossAreaPointsOfSkeletionLine, \
    removeBranchOfSkeletonLine, getBoundingBoxes, getCrossPointsOfSkeletonLine


def main():
    # target image
    target_path = "../templates/ben.png"
    target_img = cv2.imread(target_path, 0)
    _, target_img = cv2.threshold(target_img, 127, 255, cv2.THRESH_BINARY)
    print(target_img.shape)

    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)

    # independent partial structure of character
    partial_parts = getBoundingBoxes(target_img)

    print("number of partial parts: %d" % len(partial_parts))


    # skeleton line
    target_img_ = target_img != 255

    target_skel = skeletonize(target_img_)
    target_skel = (1 - target_skel) * 255

    target_skel = np.array(target_skel, dtype=np.uint8)

    for y in range(target_skel.shape[0]):
        for x in range(target_skel.shape[1]):
            if target_skel[y][x] == 0.0:
                target_img_rgb[y][x] = (0, 255, 0)

    # remove extra branch
    end_points = getEndPointsOfSkeletonLine(target_skel)
    cross_points = getCrossPointsOfSkeletonLine(target_skel)

    target_skel = removeBranchOfSkeletonLine(target_skel, end_points, cross_points)

    target_img_rgb_no_branch = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)

    for y in range(target_skel.shape[0]):
        for x in range(target_skel.shape[1]):
            if target_skel[y][x] == 0.0:
                target_img_rgb_no_branch[y][x] = (0, 255, 0)

    # new end points and cross points without extra branches
    end_points = getEndPointsOfSkeletonLine(target_skel)
    print("number of end points: %d" % len(end_points))
    cross_points = getCrossPointsOfSkeletonLine(target_skel)
    print("number of cross points: %d" % len(cross_points))

    # add end points to image with blue color
    for (x, y) in end_points:
        target_img_rgb_no_branch[y][x] = (255, 0, 0)

    # add cross points to image with red color
    for (x, y) in cross_points:
        target_img_rgb_no_branch[y][x] = (0, 0, 255)


    # Contour  of character
    target_edges = cv2.Canny(target_img, 100, 200)
    target_edges = 255 - target_edges



    cv2.imshow("img", target_img)
    cv2.imshow("skel", target_skel)
    cv2.imshow("img_rgb", target_img_rgb)
    cv2.imshow("img_rgb_no_branch", target_img_rgb_no_branch)
    cv2.imshow("edges", target_edges)

    cv2.imwrite("../templates/ben_skeleton.png", target_img_rgb)
    cv2.imwrite("../templates/ben_skeleton_no_branch.png", target_img_rgb_no_branch)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()