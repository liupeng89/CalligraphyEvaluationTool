# coding:utf-8
import cv2
import numpy as np
import math


from utils.Functions import getContourOfImage, sortPointsOnContourOfImage, removeBreakPointsOfContour, \
                            getSkeletonOfImage, removeBranchOfSkeletonLine, getEndPointsOfSkeletonLine, \
                            getCrossPointsOfSkeletonLine, getNumberOfValidPixels, splitConnectedComponents, \
                            min_distance_point2pointlist


def main():

    # 0107亻  1133壬  0554十 0427凹
    path = "0554十.jpg"

    # open image
    img = cv2.imread(path, 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # contour without break points
    contour = getContourOfImage(img.copy())
    contour = removeBreakPointsOfContour(contour)
    contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)

    contours = splitConnectedComponents(contour)
    print("contours num: %d" % len(contours))

    contours_sorted = []
    for cont in contours:
        points = sortPointsOnContourOfImage(cont)
        print("points num: %d" % len(points))
        contours_sorted.append(points)

    contour_points = []
    for y in range(contour.shape[0]):
        for x in range(contour.shape[1]):
            if contour[y][x] == 0.0:
                # black points
                contour_points.append((x, y))
    print("contour points num:%d" % len(contour_points))

    # skeleton without extra branches
    skeleton = getSkeletonOfImage(img.copy())
    # remove extra branches
    end_points = getEndPointsOfSkeletonLine(skeleton)
    cross_points = getCrossPointsOfSkeletonLine(skeleton)
    print("originale end: %d and cross: %d" % (len(end_points), len(cross_points)))
    skeleton_nobranches = removeBranchOfSkeletonLine(skeleton.copy(), end_points, cross_points)
    skeleton = skeleton_nobranches
    # new end points and cross points
    end_points = getEndPointsOfSkeletonLine(skeleton)
    cross_points = getCrossPointsOfSkeletonLine(skeleton)
    cross_points_bk = cross_points.copy()

    # merge the close points
    cross_points_merged = []
    cross_distance_threshold = 10
    used_index = []
    for i in range(len(cross_points)):
        if i in used_index:
            continue
        pt1 = cross_points[i]
        midd_pt = None
        used_index.append(i)
        for j in range(len(cross_points)):
            if i == j or j in used_index:
                continue
            pt2 = cross_points[j]

            dist = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
            if dist < cross_distance_threshold:
                used_index.append(j)
                offset = (pt1[0] - pt2[0], pt1[1] - pt2[1])
                print(offset)
                midd_pt = (pt2[0] + int(offset[0] / 2.), pt2[1] + int(offset[1] / 2.0))
                if skeleton[midd_pt[1]][midd_pt[0]] == 0.0:
                    cross_points_merged.append(midd_pt)
                else:
                    min_distance = 100000000
                    current_pt = None
                    for y in range(skeleton.shape[0]):
                        for x in range(skeleton.shape[1]):
                            if skeleton[y][x] == 0:
                                dist = math.sqrt((midd_pt[0] - x) ** 2 + (midd_pt[1] - y) ** 2)
                                if dist < min_distance:
                                    min_distance = dist
                                    current_pt = (x, y)
                    if current_pt:
                        cross_points_merged.append(current_pt)

    print("After merge cross points num: %d" % len(cross_points_merged))
    cross_points = cross_points_merged

    print("After end: %d and cross: %d" % (len(end_points), len(cross_points)))
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    # display all end points
    for pt in end_points:
        skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)
    for pt in cross_points:
        skeleton_rgb[pt[1]][pt[0]] = (0, 255, 0)
    for pt in cross_points_bk:
        skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)

    # all corner points on contour
    img = np.float32(img.copy())
    dst = cv2.cornerHarris(img, 3, 3, 0.03)
    dst = cv2.dilate(dst, None)

    corners_area_points = []
    for y in range(dst.shape[0]):
        for x in range(dst.shape[1]):
            if dst[y][x] > 0.1 * dst.max():
                corners_area_points.append((x, y))
    # show the corner points
    for pt in corners_area_points:
        if img[pt[1]][pt[0]] == 0:
            img_rgb[pt[1]][pt[0]] = (0, 255, 0)
        else:
            img_rgb[pt[1]][pt[0]] = (0, 0, 255)
    # all corner area points on the contour
    corners_lines_points = []
    for pt in corners_area_points:
        if pt in contour_points:
            corners_lines_points.append(pt)

    for pt in corners_lines_points:
        contour_rgb[pt[1]][pt[0]] = (0, 255, 0)

    # merge points of corner points
    corners_merged_points = []
    for contour_sorted in contours_sorted:
        i = 0
        while True:
            midd_index = -1
            pt = contour_sorted[i]
            if pt in corners_lines_points:
                # red point
                start = i
                end = start
                while True:
                    end += 1
                    if end >= len(contour_sorted):
                        break
                    # next point
                    next_pt = contour_sorted[end]
                    if next_pt in corners_lines_points:
                        # red point
                        continue
                    else:
                        # black point
                        break
                end -= 1
                midd_index = start + int((end - start) / 2.0)
                i = end
            i += 1
            if i >= len(contour_sorted):
                break
            if midd_index != -1:
                corners_merged_points.append(contour_sorted[midd_index])
    print("After merged, corner points num: %d" % len(corners_merged_points))
    for pt in corners_merged_points:
        contour_rgb[pt[1]][pt[0]] = (0, 0, 255)

    # remove the no-corner points
    corners_points = []
    threshold_distance = 30
    for pt in corners_merged_points:
        dist_cross = min_distance_point2pointlist(pt, cross_points)
        dist_end = min_distance_point2pointlist(pt, end_points)
        if dist_cross < threshold_distance and dist_end > threshold_distance / 3.:
            corners_points.append(pt)
    print("corner pints num: %d" % len(corners_points))
    for pt in corners_points:
        contour_rgb[pt[1]][pt[0]] = (255, 0, 0)

    # segment contour to sub-contours based on the corner points
    def segmentContourBasedOnCornerPoints(contour_sorted, corner_points):
        """
        Segment contour to sub-contours based on the corner points
        :param contour_sorted:
        :param corner_points:
        :return:
        """
        if contour_sorted is None or corner_points is None:
            return
        # sub conotour index
        sub_contour_index = []
        for pt in corner_points:
            index = contour_sorted.index(pt)
            sub_contour_index.append(index)
        print("sub contour index num: %d" % len(sub_contour_index))
        sub_contours = []
        for i in range(len(sub_contour_index)):
            if i == len(sub_contour_index) - 1:
                sub_contour = contour_sorted[sub_contour_index[i]:len(contour_sorted)] + contour_sorted[0: sub_contour_index[0] + 1]
            else:
                sub_contour = contour_sorted[sub_contour_index[i]:sub_contour_index[i + 1] + 1]
            sub_contours.append(sub_contour)
        print("sub contours num: %d" % len(sub_contours))

        return sub_contours

    # segment contour to sub-contours
    for contour in contours:
        cont_sorted = sortPointsOnContourOfImage(contour)
        sub_contours = segmentContourBasedOnCornerPoints(cont_sorted, corners_points)

    # cluster corner points
    corner_points_cluster = []
    used_index = []
    colinear_couple = []
    for i in range(len(corners_points)):
        if i in used_index:
            continue
        for j in range(len(corners_points)):
            if i == j or j in used_index:
                continue
            min_offset = min(abs(corners_points[i][0]-corners_points[j][0]), abs(corners_points[i][1]-corners_points[j][1]))
            if min_offset < 20:
                couple = [corners_points[i], corners_points[j]]
                colinear_couple.append(couple)
                used_index.append(j)
    print("co linear num: %d" % len(colinear_couple ))


    print("sub contours num: %d" % len(sub_contours))

    stroke1_img = np.ones_like(contour) * 255
    stroke1_img = np.array(stroke1_img, dtype=np.uint8)
    stroke1_img_rgb = cv2.cvtColor(stroke1_img, cv2.COLOR_GRAY2RGB)

    for pt in sub_contours[0]:
        stroke1_img_rgb[pt[1]][pt[0]] = (0, 0, 0)
        stroke1_img[pt[1]][pt[0]] = 0
    for pt in sub_contours[2]:
        stroke1_img_rgb[pt[1]][pt[0]] = (0, 0, 0)
        stroke1_img[pt[1]][pt[0]] = 0

    cv2.line(stroke1_img_rgb, sub_contours[0][0], sub_contours[2][-1], (0, 0, 255), 1)
    cv2.line(stroke1_img_rgb, sub_contours[0][-1], sub_contours[2][0], (0, 0, 255), 1)
    cv2.line(stroke1_img, sub_contours[0][0], sub_contours[2][-1], 0, 1)
    cv2.line(stroke1_img, sub_contours[0][-1], sub_contours[2][0], 0, 1)

    stroke2_img = np.ones_like(contour) * 255
    stroke2_img = np.array(stroke2_img, dtype=np.uint8)
    stroke2_img_rgb = cv2.cvtColor(stroke2_img, cv2.COLOR_GRAY2RGB)


    for pt in sub_contours[1]:
        stroke2_img_rgb[pt[1]][pt[0]] = (0, 0, 0)
        stroke2_img[pt[1]][pt[0]] = 0
    for pt in sub_contours[3]:
        stroke2_img_rgb[pt[1]][pt[0]] = (0, 0, 0)
        stroke2_img[pt[1]][pt[0]] = 0

    cv2.line(stroke2_img_rgb, sub_contours[1][0], sub_contours[3][-1], (0, 0, 255), 1)
    cv2.line(stroke2_img_rgb, sub_contours[1][-1], sub_contours[3][0], (0, 0, 255), 1)
    cv2.line(stroke2_img, sub_contours[1][0], sub_contours[3][-1], 0, 1)
    cv2.line(stroke2_img, sub_contours[1][-1], sub_contours[3][0], 0, 1)

    storke1_points = sortPointsOnContourOfImage(stroke1_img)
    stroke2_points = sortPointsOnContourOfImage(stroke2_img)

    stroke1_img = np.ones_like(stroke1_img) * 255
    stroke1_img = np.array(stroke1_img, dtype=np.uint8)

    storke1_points = np.array([storke1_points], "int32")
    cv2.fillPoly(stroke1_img, storke1_points, 0)

    stroke2_img = np.ones_like(stroke2_img) * 255
    stroke2_img = np.array(stroke2_img, dtype=np.uint8)

    storke2_points = np.array([stroke2_points], "int32")
    cv2.fillPoly(stroke2_img, storke2_points, 0)








    # find corresponding sub-contours based on the co-linear couple
    # for sub in sub_contours:
    #     pt1 = sub[0]
    #     pt2 = sub[-1]
    #
    #     couples = []
    #     for coup in colinear_couple:
    #         if pt1 in coup or pt2 in coup:
    #             # if 4 points, 2 points should be in same sub-contour
    #             if pt1 in coup and pt2 in coup:
    #                 continue
    #             couples.append(coup)
    #     print("sub couples num: %d" % len(couples))








    # cv2.imshow("img rgb", img_rgb)
    # cv2.imshow("skeleton", skeleton)
    # cv2.imshow("skeleton no branches", skeleton_nobranches )
    cv2.imshow("skeleton rgb", skeleton_rgb)
    cv2.imshow("contour rgb", contour_rgb)
    cv2.imshow("stroke 1", stroke1_img)
    cv2.imshow("stroke 2", stroke2_img)
    cv2.imshow("stroke1rgb", stroke1_img_rgb)
    cv2.imshow("stroke2rgb", stroke2_img_rgb)

    # for i in range(len(contours)):
    #     cv2.imshow("contour %d" % i, contours[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()