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
    print("After end: %d and cross: %d" % (len(end_points), len(cross_points)))
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    # display all end points
    for pt in end_points:
        skeleton_rgb[pt[1]][pt[0]] = (0, 0, 255)
    for pt in cross_points:
        skeleton_rgb[pt[1]][pt[0]] = (0, 255, 0)

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

    # cluster corner points
    corner_points_cluster = []
    used_index = []
    # for i in range(len(corners_points)):









    # cv2.imshow("img rgb", img_rgb)
    # cv2.imshow("skeleton", skeleton)
    # cv2.imshow("skeleton no branches", skeleton_nobranches )
    cv2.imshow("skeleton rgb", skeleton_rgb)
    cv2.imshow("contour rgb", contour_rgb)

    # for i in range(len(contours)):
    #     cv2.imshow("contour %d" % i, contours[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()