# coding: utf-8
import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getContourOfImage, getSkeletonOfImage, removeBreakPointsOfContour, \
                            removeBranchOfSkeletonLine, removeExtraBranchesOfSkeleton, getEndPointsOfSkeletonLine, \
                            getCrossPointsOfSkeletonLine, sortPointsOnContourOfImage, min_distance_point2pointlist, \
                            getNumberOfValidPixels, segmentContourBasedOnCornerPoints, createBlankGrayscaleImage, \
                            getLinePoints, getBreakPointsFromContour, merge_corner_lines_to_point, getCropLines, \
                            getCornerPointsOfImage, getClusterOfCornerPoints, getCropLinesPoints, \
                            getConnectedComponentsOfGrayScale, getAllMiniBoundingBoxesOfImage, getCornersPoints, \
                            getContourImage, getValidCornersPoints, getDistanceBetweenPointAndComponent, \
                            isIndependentCropLines, mergeBkAndComponent, isValidComponent


def autoStrokeExtracting(index, image, threshold_value=200):
    """
    Automatic strokes extracting
    :param image: grayscale image
    :return: strokes images with same size
    """
    strokes = []
    if image is None:
        return strokes

    # get connected components from the grayscale image, not for the binary image.
    contour_img = getContourImage(image)
    contours = getConnectedComponents(contour_img)  # no holes, num=1, holes exist, num >= 2
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

    # skeleton image of width 1 pixel of binray image
    skeleton_img = getSkeletonOfImage(img_bit)
    skeleton_img = removeExtraBranchesOfSkeleton(skeleton_img)
    end_points = getEndPointsOfSkeletonLine(skeleton_img)   # end points
    cross_points = getCrossPointsOfSkeletonLine(skeleton_img)   # croiss points

    print("end points num: %d" % len(end_points))
    print("cross points num: %d" % len(cross_points))

    if len(cross_points) == 0:
        print("no cross points!")
        strokes.append(image)
        return strokes

    # corner area points
    corners_all_points = getCornersPoints(image.copy(), contour_img, blockSize=3, ksize=3, k=0.04)
    corners_points = getValidCornersPoints(corners_all_points, cross_points, end_points, distance_threshold=30)
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
    corner_points_cluster = getClusterOfCornerPoints(corners_points, cross_points)

    # cropping lines based on the corner points
    crop_lines = getCropLines(corner_points_cluster, None)

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
        if img_[0][0] != 0.0 and isValidComponent(img_, img_bit):
            components.append(img_)

    print("components num : %d" % len(components))
    used_components = []
    component_line_relation = {}  # {component_id: [line_id1, line_id2]}

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
    for i in range(len(crop_lines)):
        components_id = []
        line = crop_lines[i]
        for j in range(len(components)):
            dist_startpt = getDistanceBetweenPointAndComponent(line[0], components[j])
            # print("dist startpt:%f" % dist_startpt)
            dist_endpt = getDistanceBetweenPointAndComponent(line[1], components[j])
            # print("dist end pt: %f" % dist_endpt)

            if dist_startpt < 3 and dist_endpt < 3:
                cv2.line(components[j], line[0], line[1], 0, 1)
                components_id.append(j)

            if len(components_id) >= 2:
                break

    # find overlap region components
    overlap_components = []
    for i in range(len(components)):
        part = components[i]
        part_lines = []
        part_lines_id = []
        for j in range(len(crop_lines)):
            line = crop_lines[j]
            if part[line[0][1]][line[0][0]] == 0.0 and part[line[1][1]][line[1][0]] == 0.0:
                part_lines.append(line)
                part_lines_id.append(j)

        # check number of lines == 4 and cross each other
        if len(part_lines) == 4:
            points_set = set()
            for line in part_lines:
                points_set.add(line[0])
                points_set.add(line[1])
            if len(points_set) == 4:
                # only 4 points
                overlap_components.append(part)
                used_components.append(i)
                component_line_relation[i] = part_lines_id

    print("overlap components num: %d" % len(overlap_components))
    print(component_line_relation)

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

        if is_single and isIndependentCropLines(part_lines):
            strokes.append(part)
            used_components.append(i)

    print("single stroke num: %d" % len(strokes))
    print("used components num: %d" % len(used_components))

    # cluster components based on the cropping lines
    for i in range(len(components)):
        if i in used_components:
            continue
        # find corresponding crop lines
        lines_id = []
        for j in range(len(crop_lines)):
            line = crop_lines[j]
            if components[i][line[0][1]][line[0][0]] == 0.0 and components[i][line[1][1]][line[1][0]] != 0.0:
                lines_id.append(j)
            if components[i][line[0][1]][line[0][0]] != 0.0 and components[i][line[1][1]][line[1][0]] == 0.0:
                lines_id.append(j)

        if len(lines_id) == 0:
            continue
        component_line_relation[i] = lines_id
        used_components.append(i)

    # cluster components based on the relations and merge those related components
    clusters = []
    for k1, v1 in component_line_relation.items():

        cluster = [k1]; value_sets = [set(v1)]

        for k2, v2 in component_line_relation.items():
            is_related = True
            for value in value_sets:
                if not value.intersection(set(v2)):
                    is_related = False
                    break
            if is_related and k2 not in cluster:
                cluster.append(k2)
                value_sets.append(set(v2))
        cluster = sorted(cluster)
        if cluster not in clusters:
            clusters.append(cluster)

    print(clusters)

    # merge components based on the cluster
    for i in range(len(clusters)):
        cluster = clusters[i]
        bk = createBlankGrayscaleImage(image)

        for clt in cluster:
            bk = mergeBkAndComponent(bk, components[clt])

        # add to strokes
        strokes.append(bk)

    # check the stroke is valid
    for i in range(len(strokes)):
        stroke = strokes[i]

    cv2.imshow("radical_%d" % index, contour_rgb)
    cv2.imshow("radical_gray_%d" % index, contour_gray)

    return strokes


def main():
    # 1133壬 2252支 0631叟 0633口 0242俄 0195佛 0860善 0059乘 0098亩 0034串
    path = "0034串.jpg"

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

    # display strokes
    for i in range(len(total_strokes)):
        cv2.imshow("stroke_%d"%i, total_strokes[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()