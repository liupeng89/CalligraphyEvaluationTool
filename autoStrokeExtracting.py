# coding: utf-8
import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeBranchOfSkeleton, getEndPointsOfSkeletonLine, \
                          getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints, createBlankGrayscaleImage, \
                            getLinePoints, getBreakPointsFromContour, merge_corner_lines_to_point, getCropLines, \
                            getCornerPointsOfImage, getClusterOfCornerPoints, getCropLinesPoints, \
                            getConnectedComponentsOfGrayScale, getAllMiniBoundingBoxesOfImage, getCornersPoints, \
                            getContourImage, getValidCornersPoints


def autoStrokeExtracting(index, image, threshold_value=200):
    """
    Automatic strokes extracting
    :param image: grayscale image
    :return: strokes images with same size
    """
    strokes = []
    if image is None:
        return strokes

    # get connected components
    contour_img = getContourImage(image)
    contours = getConnectedComponents(contour_img)
    print("contours num: %d" % len(contours))

    corners_points_sorted = []
    for ct in contours:
        points = sortPointsOnContourOfImage(ct)
        corners_points_sorted.append(points)
    if len(corners_points_sorted) == 1:
        print("No holes exist!")
    elif len(corners_points_sorted) >= 2:
        print("Holes exist!")

    # grayscale image to binary image
    _, img_bit = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # skeleton
    skeleton_img = getSkeletonOfImage(img_bit)
    end_points = getEndPointsOfSkeletonLine(skeleton_img)
    cross_points = getCrossPointsOfSkeletonLine(skeleton_img)

    print("end points num: %d" % len(end_points))
    print("cross points num: %d" % len(cross_points))

    if len(cross_points) == 0:
        print("no cross points!")
        strokes.append(image)
        return strokes

    # corner area points
    corners_all_points = getCornersPoints(image.copy(), contour_img)
    corners_points = getValidCornersPoints(corners_all_points, cross_points, end_points)
    print("corners points num: %d" % len(corners_points))


    contour_rgb = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)
    for pt in corners_points:
        contour_rgb[pt[1]][pt[0]] = (0, 0, 255)

    # cluster corners points based on the cross point
    dist_threshold = 30
    corner_points_cluster = getClusterOfCornerPoints(corners_points, cross_points)

    crop_lines = getCropLines(corner_points_cluster)
    for line in crop_lines:
        cv2.line(contour_rgb, line[0], line[1], (0, 255, 0), 1)


    cv2.imshow("radical_%d" % index, contour_rgb)

    return strokes


def main():
    # 1133壬 2252支 0631叟 0633口 0242俄 0195佛
    path = "0195佛.jpg"

    img_rgb = cv2.imread(path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, img_bit = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    # grayscale threshold filter pixel > 200 = 255
    for y in range(img_gray.shape[0]):
        for x in range(img_gray.shape[1]):
            if img_gray[y][x] > 200:
                img_gray[y][x] = 255

    # get components
    components = getConnectedComponentsOfGrayScale(img_gray, threshold_value=200)

    if components is None or len(components) == 0:
        print("components num is 0")
        return

    total_strokes = []

    for i in range(len(components)):
        radical = components[i]

        radical_strokes = autoStrokeExtracting(i, radical)
        if radical_strokes is None or len(radical_strokes) == 0:
            print("radiacal is None")
        else:
            total_strokes += radical_strokes

    # cv2.imshow("img gray", img_gray)
    #
    # for i in range(len(components)):
    #     cv2.imshow("stroke_%d"%i, components[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()