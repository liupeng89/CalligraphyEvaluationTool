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
                            getContourImage, getValidCornersPoints, getDistanceBetweenPointAndComponent, \
                            isIndependentCropLines


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

    if len(corners_points) == 0:
        print("no corner point")
        strokes.append(image)
        return strokes

    contour_rgb = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)
    contour_gray = cv2.cvtColor(contour_rgb, cv2.COLOR_RGB2GRAY)
    _, contour_gray = cv2.threshold(contour_gray, 200, 255, cv2.THRESH_BINARY)

    for pt in corners_points:
        contour_rgb[pt[1]][pt[0]] = (0, 0, 255)

    # cluster corners points based on the cross point
    dist_threshold = 30
    corner_points_cluster = getClusterOfCornerPoints(corners_points, cross_points)

    crop_lines = getCropLines(corner_points_cluster)
    for line in crop_lines:
        cv2.line(contour_rgb, line[0], line[1], (0, 255, 0), 1)
        cv2.line(contour_gray, line[0], line[1], 0, 1)

    # split contour to components
    ret, labels = cv2.connectedComponents(contour_gray, connectivity=4)
    components = []
    for r in range(1, ret):
        img_ = createBlankGrayscaleImage(contour_gray)
        for y in range(contour_gray.shape[0]):
            for x in range(contour_gray.shape[1]):
                if labels[y][x] == r:
                    img_[y][x] = 0.0
        if img_[0][0] != 0.0:
            components.append(img_)

    print("components num : %d" % len(components))
    used_components = []

    # merge contour to components
    for i in range(len(components)):
        merge_points = []
        for y in range(1, contour_gray.shape[0]-1):
            for x in range(1, contour_gray.shape[1]-1):
                if contour_gray[y][x] == 0.0:
                    # 4 nearby pixels should be black in components
                    if components[i][y-1][x] == 0.0 or components[i][y][x+1] == 0.0 or components[i][y+1][x] == 0.0 or \
                            components[i][y][x-1] == 0.0:
                        merge_points.append((x, y))
        for pt in merge_points:
            components[i][pt[1]][pt[0]] = 0.0
    # merge cropping lines on the components
    for i in range(len(components)):
        for line in crop_lines:

            dist_startpt = getDistanceBetweenPointAndComponent(line[0], components[i])
            print("dist startpt:%f" % dist_startpt)
            dist_endpt = getDistanceBetweenPointAndComponent(line[1], components[i])
            print("dist end pt: %f" % dist_endpt)

            if dist_startpt < 3 and dist_endpt < 3:
                cv2.line(components[i], line[0], line[1], 0, 1)

    # find overlap region components
    overlap_components = []
    for i in range(len(components)):
        part = components[i]
        part_lines = []
        for line in crop_lines:
            if part[line[0][1]][line[0][0]] == 0.0 and part[line[1][1]][line[1][0]] == 0.0:
                part_lines.append(line)
        # check number of lines == 4 and cross each other
        if len(part_lines) == 4:
            pass






    # cluster components based on the cropping lines
    for i in range(len(components)):
        part = components[i]

        part_lines = [] # used to detect overlap region components.

        # find single part is stroke
        is_single = True
        for line in crop_lines:
            x1 = line[0][0]; y1 = line[0][1]
            x2 = line[1][0]; y2 = line[1][1]
            if part[y1][x1] == 0.0 and part[y2][x2] != 0.0 or part[y1][x1] != 0.0 and part[y2][x2] == 0.0:
                is_single = False
                break
            if part[y1][x1] == 0.0 and part[y2][x2] == 0.0:
                part_lines.append(line)
        print("part lines num: %d" % len(part_lines))
        if is_single and isIndependentCropLines(part_lines):
            strokes.append(part)
            used_components.append(i)

    print("single stroke num: %d" % len(strokes))
    print("used components num: %d" % len(used_components))
    print(used_components)

    # cluster components based on the cropping lines




    cv2.imshow("radical_%d" % index, contour_rgb)
    cv2.imshow("radical_gray_%d" % index, contour_gray)

    for i in range(len(components)):
        cv2.imshow("ra_%d_com_%d" % (index, i), components[i])

    return strokes


def main():
    # 1133壬 2252支 0631叟 0633口 0242俄 0195佛 0860善
    path = "0631叟.jpg"

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
    for i in range(len(total_strokes)):
        cv2.imshow("stroke_%d"%i, total_strokes[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()