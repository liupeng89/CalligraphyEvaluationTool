import cv2
import numpy as np
import math

from utils.Functions import getConnectedComponents, getSkeletonOfImage, getEndPointsOfSkeletonLine, \
                            getCrossPointsOfSkeletonLine, createBlankGrayscaleImage, getCropLines, \
                            getClusterOfCornerPoints, getAllMiniBoundingBoxesOfImage, getContourImage, \
                            getValidCornersPoints, getDistanceBetweenPointAndComponent, isValidComponent, \
                            removeShortBranchesOfSkeleton, sortPointsOnContourOfImage, removeBreakPointsOfContour
from algorithms.RDP import rdp


def autoStrokeExtractFromComponent(component):
    """
    Automatically strokes extract from the component.
    :param component:
    :return:
    """
    strokes = []
    if component is None:
        return strokes

    # 6. Get skeletons of component.
    comp_skeleton = getSkeletonOfImage(component.copy())
    # cv2.imshow("skeleton_original", comp_skeleton)

    # 7. Process the skeleton by remove extra branches.
    comp_skeleton = removeShortBranchesOfSkeleton(comp_skeleton, length_threshold=30)
    # cv2.imshow("skeleton_smoothed", comp_skeleton)

    # 8. Get the end points and cross points after skeleton processed
    end_points = getEndPointsOfSkeletonLine(comp_skeleton)
    cross_points = getCrossPointsOfSkeletonLine(comp_skeleton)
    print("end points num: %d ,and cross points num: %d" % (len(end_points), len(cross_points)))

    # 9. Get contour image of component
    comp_contours_img = getContourImage(component.copy())

    # 10. Detect the number of contours and return all contours
    comp_contours = getConnectedComponents(comp_contours_img, connectivity=8)
    print("contours num: %d" % len(comp_contours))

    # 11. Get points on contours
    corners_points = []
    for cont in comp_contours:
        cont = removeBreakPointsOfContour(cont)
        cont_sorted = sortPointsOnContourOfImage(cont)
        cont_points = rdp(cont_sorted, 5)
        corners_points += cont_points
    print("corner points num:", len(corners_points))

    CORNER_CROSS_DIST_THRESHOLD = 30
    corners_points_merged = []
    for pt in corners_points:
        for cpt in cross_points:
            dist = math.sqrt((pt[0] - cpt[0]) ** 2 + (pt[1] - cpt[1]) ** 2)
            if dist < CORNER_CROSS_DIST_THRESHOLD:
                corners_points_merged.append(pt)
                break
    corners_points = corners_points_merged
    print("merged corner points num:", len(corners_points))

    # # 11. Detect the corner regions of component
    # #       Harris corner detector
    # corner_region_img = np.float32(component.copy())
    # dst = cv2.cornerHarris(corner_region_img, blockSize=3, ksize=3, k=0.04)
    # dst = cv2.dilate(dst, None)
    #
    # # get all points in corners area
    # corners_area_points = []
    # for y in range(dst.shape[0]):
    #     for x in range(dst.shape[1]):
    #         if dst[y][x] > 0.1 * dst.max():
    #             corners_area_points.append((x, y))
    #
    # # get all center points of corner area
    # corners_img = createBlankGrayscaleImage(component)
    # for pt in corners_area_points:
    #     corners_img[pt[1]][pt[0]] = 0.0
    #
    # rectangles = getAllMiniBoundingBoxesOfImage(corners_img)
    #
    # corners_area_center_points = []
    # for rect in rectangles:
    #     corners_area_center_points.append((rect[0] + int(rect[2] / 2.), rect[1] + int(rect[3] / 2.)))
    #
    # # get all corner points in coutour image.
    # corners_points = []
    # for pt in corners_area_center_points:
    #     if comp_contours_img[pt[1]][pt[0]] == 0.0:
    #         corners_points.append(pt)
    #     else:
    #         min_dist = 100000
    #         min_x = min_y = 0
    #         for y in range(comp_contours_img.shape[0]):
    #             for x in range(comp_contours_img.shape[1]):
    #                 cpt = comp_contours_img[y][x]
    #                 if cpt == 0.0:
    #                     dist = math.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
    #                     if dist < min_dist:
    #                         min_dist = dist
    #                         min_x = x
    #                         min_y = y
    #         # points on contour
    #         corners_points.append((min_x, min_y))
    print("corners points num: %d" % len(corners_points))

    # 12. Get valid corner points based on the end points and cross points
    corners_points = getValidCornersPoints(corners_points, cross_points, end_points, distance_threshold=30)
    print("corners points num: %d" % len(corners_points))

    if len(corners_points) == 0:
        print("no corner points")
        strokes.append(component)
        return strokes

    # 13. Cluster these corner points based on the distance between them and cross points
    corners_points_cluster = getClusterOfCornerPoints(corners_points, cross_points, threshold_distance=70)
    print("corner points cluster num: %d" % len(corners_points_cluster))
    print(corners_points_cluster)

    # 14. Generate cropping lines between two corner points
    crop_lines = getCropLines(corners_points_cluster, comp_contours)
    print("cropping lines num: %d" % len(crop_lines))

    # 15. Separate the components based on the cropping lines
    component_ = component.copy()

    # add white and 1-pixel width line in component to separate it.
    for line in crop_lines:
        cv2.line(component_, line[0], line[1], 255, 1)

    # 16. Get parts of component.
    comp_parts = []
    # invert color !!!
    component_ = 255 - component_
    ret, labels = cv2.connectedComponents(component_, connectivity=4)
    print(ret)
    for r in range(1, ret):
        img_ = createBlankGrayscaleImage(component_)
        for y in range(component_.shape[0]):
            for x in range(component_.shape[1]):
                if labels[y][x] == r:
                    img_[y][x] = 0.0
        if img_[0][0] != 0.0 and isValidComponent(img_, component):
            comp_parts.append(img_)
    print("parts of component num: %d" % len(comp_parts))

    # 17. Add cropping lines to corresponding parts of component
    # add lines to parts of component.
    for i in range(len(comp_parts)):
        part = comp_parts[i]

        for line in crop_lines:
            start_dist = getDistanceBetweenPointAndComponent(line[0], part)
            end_dist = getDistanceBetweenPointAndComponent(line[1], part)

            if start_dist <= 3 and end_dist <= 3:
                cv2.line(part, line[0], line[1], 0, 1)

    # 18. Find intersection parts of component
    used_index = []
    intersect_parts_index = []
    for i in range(len(comp_parts)):
        part = comp_parts[i]
        num = 0  # number of lines in part
        for line in crop_lines:
            if part[line[0][1]][line[0][0]] == 0.0 and part[line[1][1]][line[1][0]] == 0.0:
                num += 1
        if num == 4:   # 4 lines in one part, this part is the intersection part
            intersect_parts_index.append(i)
            used_index.append(i)
    print("intersection parts num: %d" % len(intersect_parts_index))

    # 19. Find the relation part and crop lines - one line -> one part or three part (part1 + intersect_part + part2)
    intersect_parts_crop_lines_index = []
    for i in range(len(crop_lines)):
        line = crop_lines[i]
        for index in intersect_parts_index:
            intersect_part = comp_parts[index]
            if intersect_part[line[0][1]][line[0][0]] == 0.0 and intersect_part[line[1][1]][line[1][0]] == 0.0:
                # this line in intersection part
                intersect_parts_crop_lines_index.append(i)
    print("crop lines in intersection part num: %d" % len(intersect_parts_crop_lines_index))

    # Cropping lines are divided into two types: in intersect part and not in this part.
    line_parts_relation = []
    for index in intersect_parts_crop_lines_index:
        line = crop_lines[index]

        # line and parts that are connected by this crop line: A - intersect_part - B
        line_connected_parts = []
        # find intersection part contains this line
        for i in intersect_parts_index:
            intersect_part = comp_parts[i]
            if intersect_part[line[0][1]][line[0][0]] == 0.0 and intersect_part[line[1][1]][line[1][0]] == 0.0:
                # line in this intersect part
                line_connected_parts.append(i)
        # find two parts connectd by this crop line
        for i in range(len(comp_parts)):

            # part should not be the intersect part
            if i in intersect_parts_index:
                continue

            # check only end point of line in part.
            part = comp_parts[i]
            if part[line[0][1]][line[0][0]] == 0.0 and part[line[1][1]][line[1][0]] != 0.0 or \
                    part[line[0][1]][line[0][0]] != 0.0 and part[line[1][1]][line[1][0]] == 0.0:
                line_connected_parts.append(i)

        # add line connected parts to relation list.
        if line_connected_parts not in line_parts_relation:
            line_parts_relation.append(line_connected_parts)

    # add independent parts to relation of line and parts
    for i in range(len(comp_parts)):
        line_connected_parts = []
        is_independent = True
        for relation in line_parts_relation:
            if i in relation:
                is_independent = False
        # check this part is independent or not
        if is_independent:
            line_connected_parts.append(i)

        if line_connected_parts != []:
            line_parts_relation.append(line_connected_parts)

    # 20. Merge parts based on the line parts relation
    for i in range(len(line_parts_relation)):
        # blank image
        blank_ = createBlankGrayscaleImage(component)
        # parts relation list
        relation = line_parts_relation[i]

        for rel in relation:
            part = comp_parts[rel]
            if part is None:
                continue
            # merge part and blank image
            for y in range(part.shape[0]):
                for x in range(part.shape[1]):
                    if part[y][x] == 0.0:
                        blank_[y][x] = 0.0

        # add to strokes list
        strokes.append(blank_)
    return strokes


def autoStrokeExtractFromCharacter(rgb):
    """
    Automatically stroke extract.
    :param rgb: rbg image
    :return: strokes images
    """
    if rgb is None:
        return

    # 1. Load RGB image;
    # 2. Convert RGB image to gryascale image;
    img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 3. Convert grayscale to binary image;
    _, img_bit = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 4. Get components from binary image;
    components = getConnectedComponents(img_bit, connectivity=8)
    print("components num: %d" % len(components))

    # 5. Process each component;
    strokes = []
    for comp in components:
        sub_strokes = autoStrokeExtractFromComponent(comp)
        print("sub_strokes num: %d" % len(sub_strokes))
        strokes += sub_strokes
        break
    return strokes

#
# def autoStrokeExtractFromComponent(component):
#     """
#     Automatically strokes extract from the component.
#     :param component:
#     :return:
#     """
#     strokes = []
#     if component is None:
#         return strokes
#
#     # 6. Get skeletons of component.
#     comp_skeleton = getSkeletonOfImage(component.copy())
#     # cv2.imshow("skeleton_original", comp_skeleton)
#
#     # 7. Process the skeleton by remove extra branches.
#     comp_skeleton = removeShortBranchesOfSkeleton(comp_skeleton, length_threshold=30)
#     # cv2.imshow("skeleton_smoothed", comp_skeleton)
#
#     # 8. Get the end points and cross points after skeleton processed
#     end_points = getEndPointsOfSkeletonLine(comp_skeleton)
#     cross_points = getCrossPointsOfSkeletonLine(comp_skeleton)
#     print("end points num: %d ,and cross points num: %d" % (len(end_points), len(cross_points)))
#
#     # 9. Get contour image of component
#     comp_contours_img = getContourImage(component.copy())
#
#     # 10. Detect the number of contours and return all contours
#     comp_contours = getConnectedComponents(comp_contours_img, connectivity=8)
#     print("contours num: %d" % len(comp_contours))
#
#     # 11. Detect the corner regions of component
#     #       Harris corner detector
#     corner_region_img = np.float32(component.copy())
#     dst = cv2.cornerHarris(corner_region_img, blockSize=3, ksize=3, k=0.04)
#     dst = cv2.dilate(dst, None)
#
#     # get all points in corners area
#     corners_area_points = []
#     for y in range(dst.shape[0]):
#         for x in range(dst.shape[1]):
#             if dst[y][x] > 0.1 * dst.max():
#                 corners_area_points.append((x, y))
#
#     # get all center points of corner area
#     corners_img = createBlankGrayscaleImage(component)
#     for pt in corners_area_points:
#         corners_img[pt[1]][pt[0]] = 0.0
#
#     rectangles = getAllMiniBoundingBoxesOfImage(corners_img)
#
#     corners_area_center_points = []
#     for rect in rectangles:
#         corners_area_center_points.append((rect[0] + int(rect[2] / 2.), rect[1] + int(rect[3] / 2.)))
#
#     # get all corner points in coutour image.
#     corners_points = []
#     for pt in corners_area_center_points:
#         if comp_contours_img[pt[1]][pt[0]] == 0.0:
#             corners_points.append(pt)
#         else:
#             min_dist = 100000
#             min_x = min_y = 0
#             for y in range(comp_contours_img.shape[0]):
#                 for x in range(comp_contours_img.shape[1]):
#                     cpt = comp_contours_img[y][x]
#                     if cpt == 0.0:
#                         dist = math.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
#                         if dist < min_dist:
#                             min_dist = dist
#                             min_x = x
#                             min_y = y
#             # points on contour
#             corners_points.append((min_x, min_y))
#     print("corners points num: %d" % len(corners_points))
#
#     # 12. Get valid corner points based on the end points and cross points
#     corners_points = getValidCornersPoints(corners_points, cross_points, end_points, distance_threshold=30)
#     print("corners points num: %d" % len(corners_points))
#
#     if len(corners_points) == 0:
#         print("no corner points")
#         strokes.append(component)
#         return strokes
#
#     # 13. Cluster these corner points based on the distance between them and cross points
#     corners_points_cluster = getClusterOfCornerPoints(corners_points, cross_points)
#     print("corner points cluster num: %d" % len(corners_points_cluster))
#
#     # 14. Generate cropping lines between two corner points
#     crop_lines = getCropLines(corners_points_cluster, None)
#     print("cropping lines num: %d" % len(crop_lines))
#
#     # 15. Separate the components based on the cropping lines
#     component_ = component.copy()
#
#     # add white and 1-pixel width line in component to separate it.
#     for line in crop_lines:
#         cv2.line(component_, line[0], line[1], 255, 1)
#
#     # 16. Get parts of component.
#     comp_parts = []
#     # invert color !!!
#     component_ = 255 - component_
#     ret, labels = cv2.connectedComponents(component_, connectivity=4)
#     print(ret)
#     for r in range(1, ret):
#         img_ = createBlankGrayscaleImage(component_)
#         for y in range(component_.shape[0]):
#             for x in range(component_.shape[1]):
#                 if labels[y][x] == r:
#                     img_[y][x] = 0.0
#         if img_[0][0] != 0.0 and isValidComponent(img_, component):
#             comp_parts.append(img_)
#     print("parts of component num: %d" % len(comp_parts))
#
#     # 17. Add cropping lines to corresponding parts of component
#     # add lines to parts of component.
#     for i in range(len(comp_parts)):
#         part = comp_parts[i]
#
#         for line in crop_lines:
#             start_dist = getDistanceBetweenPointAndComponent(line[0], part)
#             end_dist = getDistanceBetweenPointAndComponent(line[1], part)
#
#             if start_dist <= 3 and end_dist <= 3:
#                 cv2.line(part, line[0], line[1], 0, 1)
#
#     # 18. Find intersection parts of component
#     used_index = []
#     intersect_parts_index = []
#     for i in range(len(comp_parts)):
#         part = comp_parts[i]
#         num = 0  # number of lines in part
#         for line in crop_lines:
#             if part[line[0][1]][line[0][0]] == 0.0 and part[line[1][1]][line[1][0]] == 0.0:
#                 num += 1
#         if num == 4:   # 4 lines in one part, this part is the intersection part
#             intersect_parts_index.append(i)
#             used_index.append(i)
#     print("intersection parts num: %d" % len(intersect_parts_index))
#
#     # 19. Find the relation part and crop lines - one line -> one part or three part (part1 + intersect_part + part2)
#     intersect_parts_crop_lines_index = []
#     for i in range(len(crop_lines)):
#         line = crop_lines[i]
#         for index in intersect_parts_index:
#             intersect_part = comp_parts[index]
#             if intersect_part[line[0][1]][line[0][0]] == 0.0 and intersect_part[line[1][1]][line[1][0]] == 0.0:
#                 # this line in intersection part
#                 intersect_parts_crop_lines_index.append(i)
#     print("crop lines in intersection part num: %d" % len(intersect_parts_crop_lines_index))
#
#     # Cropping lines are divided into two types: in intersect part and not in this part.
#     line_parts_relation = []
#     for index in intersect_parts_crop_lines_index:
#         line = crop_lines[index]
#
#         # line and parts that are connected by this crop line: A - intersect_part - B
#         line_connected_parts = []
#         # find intersection part contains this line
#         for i in intersect_parts_index:
#             intersect_part = comp_parts[i]
#             if intersect_part[line[0][1]][line[0][0]] == 0.0 and intersect_part[line[1][1]][line[1][0]] == 0.0:
#                 # line in this intersect part
#                 line_connected_parts.append(i)
#         # find two parts connectd by this crop line
#         for i in range(len(comp_parts)):
#
#             # part should not be the intersect part
#             if i in intersect_parts_index:
#                 continue
#
#             # check only end point of line in part.
#             part = comp_parts[i]
#             if part[line[0][1]][line[0][0]] == 0.0 and part[line[1][1]][line[1][0]] != 0.0 or \
#                     part[line[0][1]][line[0][0]] != 0.0 and part[line[1][1]][line[1][0]] == 0.0:
#                 line_connected_parts.append(i)
#
#         # add line connected parts to relation list.
#         if line_connected_parts not in line_parts_relation:
#             line_parts_relation.append(line_connected_parts)
#
#     # add independent parts to relation of line and parts
#     for i in range(len(comp_parts)):
#         line_connected_parts = []
#         is_independent = True
#         for relation in line_parts_relation:
#             if i in relation:
#                 is_independent = False
#         # check this part is independent or not
#         if is_independent:
#             line_connected_parts.append(i)
#
#         if line_connected_parts != []:
#             line_parts_relation.append(line_connected_parts)
#
#     # 20. Merge parts based on the line parts relation
#     for i in range(len(line_parts_relation)):
#         # blank image
#         blank_ = createBlankGrayscaleImage(component)
#         # parts relation list
#         relation = line_parts_relation[i]
#
#         for rel in relation:
#             part = comp_parts[rel]
#             if part is None:
#                 continue
#             # merge part and blank image
#             for y in range(part.shape[0]):
#                 for x in range(part.shape[1]):
#                     if part[y][x] == 0.0:
#                         blank_[y][x] = 0.0
#
#         # add to strokes list
#         strokes.append(blank_)
#     return strokes
#
#
# def autoStrokeExtractFromCharacter(rgb):
#     """
#     Automatically stroke extract.
#     :param rgb: rbg image
#     :return: strokes images
#     """
#     if rgb is None:
#         return
#
#     # 1. Load RGB image;
#     # 2. Convert RGB image to gryascale image;
#     img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#
#     # 3. Convert grayscale to binary image;
#     _, img_bit = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
#
#     # 4. Get components from binary image;
#     components = getConnectedComponents(img_bit, connectivity=8)
#     print("components num: %d" % len(components))
#
#     # 5. Process each component;
#     strokes = []
#     for comp in components:
#         sub_strokes = autoStrokeExtractFromComponent(comp)
#         print("sub_strokes num: %d" % len(sub_strokes))
#         strokes += sub_strokes
#
#     return strokes