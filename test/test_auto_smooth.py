import cv2
import numpy as np
import math

from utils.Functions import createBlankGrayscaleImage, getAllMiniBoundingBoxesOfImage, getContourOfImage, \
                            getConnectedComponents, removeBreakPointsOfContour, sortPointsOnContourOfImage, \
                            fitCurve, draw_cubic_bezier, createBlankRGBImage, getSkeletonOfImage, \
                            removeBranchOfSkeletonLine, getEndPointsOfSkeletonLine, getCrossPointsOfSkeletonLine


def autoSmoothContoursOfComponent(component, blockSize=3, ksize=3, k=0.04):
    """

    :param component:
    :return:
    """
    if component is None:
        return
    # 5. Using corner detection to get corner regions
    corner_component = np.float32(component)
    dst = cv2.cornerHarris(corner_component, blockSize=blockSize, ksize=ksize, k=k)
    dst = cv2.dilate(dst, None)

    corners_area_points = []
    for y in range(dst.shape[0]):
        for x in range(dst.shape[1]):
            if dst[y][x] > 0.1 * dst.max():
                corners_area_points.append((x, y))
    print("corner area points num: %d" % len(corners_area_points))

    # 6. Determine center points of corner areas
    blank_gray = createBlankGrayscaleImage(component)
    for pt in corners_area_points:
        blank_gray[pt[1]][pt[0]] = 0.0

    rectangles = getAllMiniBoundingBoxesOfImage(blank_gray)

    corners_area_center_points = []
    for rect in rectangles:
        corners_area_center_points.append((rect[0] + int(rect[2] / 2.), rect[1] + int(rect[3] / 2.)))
    print("corner area center points num: %d" % len(corners_area_center_points))

    # based the distance to end points and cross points, remove extra corners area center points

    component_skeleton = getSkeletonOfImage(component)
    end_points = getEndPointsOfSkeletonLine(component_skeleton)
    cross_points = getCrossPointsOfSkeletonLine(component_skeleton)

    # remove extra branches
    # img_skeleton = removeBranchOfSkeletonLine(img_skeleton, end_points, cross_points)
    # end_points = getEndPointsOfSkeletonLine(img_skeleton)
    # cross_points = getEndPointsOfSkeletonLine(img_skeleton)

    # detect valid corner region center points closed to end points and cross points
    valid_corners_area_center_points = []
    dist_threshold = 40
    for pt in corners_area_center_points:
        is_valid = False
        for ept in end_points:
            dist = math.sqrt((pt[0] - ept[0]) ** 2 + (pt[1] + ept[1]) ** 2)
            if dist <= dist_threshold:
                is_valid = True
                break
        if is_valid:
            valid_corners_area_center_points.append(pt)
            continue
        for cpt in cross_points:
            dist = math.sqrt((pt[0] - cpt[0]) ** 2 + (pt[1] - cpt[1]) ** 2)
            if dist <= dist_threshold:
                is_valid = True
                break
        if is_valid:
            valid_corners_area_center_points.append(pt)

    print("valid corner area center points num: %d" % len(valid_corners_area_center_points))

    del blank_gray

    # 7. Get all contours of component
    component_contours = getContourOfImage(component)
    contours = getConnectedComponents(component_contours, connectivity=8)
    print("contours num: %d" % len(contours))

    # 8. Process contours to get closed and 1-pixel width contours
    contours_processed = []
    for cont in contours:
        cont = removeBreakPointsOfContour(cont)
        contours_processed.append(cont)
    print("contours processed num: %d" % len(contours_processed))

    # 9. Find corner points of conthours closed to corner region center points. For each contour, there is a coner points list.
    contours_corner_points = []
    for i in range(len(contours_processed)):
        corner_points = []
        contour = contours_processed[i]

        for pt in valid_corners_area_center_points:
            x0 = target_x = pt[0];
            y0 = target_y = pt[1]
            min_dist = 10000
            # search target point in region: 20 * 20 of center is (x0, y0)
            for y in range(y0 - 10, y0 + 10):
                for x in range(x0 - 10, x0 + 10):
                    if contour[y][x] == 255:
                        continue
                    dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                    if dist < min_dist:
                        min_dist = dist;
                        target_x = x;
                        target_y = y
            if min_dist < 5:
                corner_points.append((target_x, target_y))

        contours_corner_points.append(corner_points)
    total_num = 0
    for cont in contours_corner_points:
        total_num += len(cont)
    if total_num == len(valid_corners_area_center_points):
        print("corner points not ignored")
    else:
        print("corner points be ignored")

    # 10. Separate contours into sub-contours based on the corner points on different contours
    sub_contours = []
    for i in range(len(contours_processed)):
        contour = contours_processed[i]
        corner_points = contours_corner_points[i]
        # sorted the contour
        contour_points_sorted = sortPointsOnContourOfImage(contour)
        # sorted the corner points
        corner_points_sorted = []
        for pt in contour_points_sorted:
            if pt in corner_points:
                corner_points_sorted.append(pt)
        # sepate the contour into sub-contour
        for j in range(len(corner_points_sorted)):
            start_pt = corner_points_sorted[j]
            end_pt = None
            if j == len(corner_points_sorted) - 1:
                end_pt = corner_points_sorted[0]
            else:
                end_pt = corner_points_sorted[j + 1]
            # find indexes of start point and end point in contour_points_sorted
            start_index = contour_points_sorted.index(start_pt)
            end_index = contour_points_sorted.index(end_pt)

            # separate
            sub_contour = None
            if start_index <= end_index:
                if end_index == len(contour_points_sorted) - 1:
                    sub_contour = contour_points_sorted[start_index: len(contour_points_sorted)]
                    sub_contour.append(contour_points_sorted[0])
                else:
                    sub_contour = contour_points_sorted[start_index: end_index + 1]
            else:
                sub_contour = contour_points_sorted[start_index: len(contour_points_sorted)] + contour_points_sorted[
                                                                                               0: end_index + 1]

            sub_contours.append(sub_contour)
    print("sub contours num: %d" % len(sub_contours))

    # 11. Beizer curve fit all sub-contours under maximal error
    max_error = 100
    sub_contours_smoothed = []

    for id in range(len(sub_contours)):
        # single sub-contour
        sub_contour = np.array(sub_contours[id])

        if len(sub_contour) < 2:
            continue
        beziers = fitCurve(sub_contour, maxError=max_error)
        sub_contour_smoothed = []

        for bez in beziers:
            bezier_points = draw_cubic_bezier(bez[0], bez[1], bez[2], bez[3])
            sub_contour_smoothed += bezier_points

        sub_contours_smoothed.append(sub_contour_smoothed)

    # 12. Merge sub-contours together
    img_smoothed_gray = createBlankGrayscaleImage(component)

    # merge all smoothed sub-contours
    for sub in sub_contours_smoothed:
        for pt in sub:
            img_smoothed_gray[pt[1]][pt[0]] = 0.0
    # process smoothed contours to get closed and 1-pixel width
    img_smoothed_gray = getSkeletonOfImage(img_smoothed_gray)

    # remove single points that 8

    cv2.imshow("img_smoothed_gray", img_smoothed_gray)

    contours_smoothed = getConnectedComponents(img_smoothed_gray)

    if len(contours_smoothed) == 1:
        # no hole exist, directly fill black in the contour
        cont = contours_smoothed[0]
        cont_points = sortPointsOnContourOfImage(cont)
        cont_points = np.array([cont_points], "int32")

        fill_contour_smooth = np.ones_like(component) * 255
        fill_contour_smooth = np.array(fill_contour_smooth, dtype=np.uint8)
        fill_contour_smooth = cv2.fillPoly(fill_contour_smooth, cont_points, 0)

        return fill_contour_smooth
    else:
        # exist hole, should processed
        print("there are holes!")
        fill_img_list = []
        hole_points = []
        for cont in contours_smoothed:
            cont_points = sortPointsOnContourOfImage(cont)
            cont_points = np.array([cont_points], "int32")

            fill_contour_smooth = np.ones_like(component) * 255
            fill_contour_smooth = np.array(fill_contour_smooth, dtype=np.uint8)
            fill_contour_smooth = cv2.fillPoly(fill_contour_smooth, cont_points, 0)

            valid_num = same_num = 0
            for y in range(component.shape[0]):
                for x in range(component.shape[1]):
                    if component[y][x] == 0.0:
                        valid_num += 1
                        if fill_contour_smooth[y][x] == 0.0:
                            same_num += 1

            if 1.0 * same_num / valid_num > 0.8:
                fill_img_list.append(fill_contour_smooth)
                print("ratio: %f" % (1.0 * same_num / valid_num))
            else:
                print("ratio: %f" % (1.0 * same_num / valid_num))
                for y in range(fill_contour_smooth.shape[0]):
                    for x in range(fill_contour_smooth.shape[1]):
                        if fill_contour_smooth[y][x] == 0.0:
                            hole_points.append((x, y))

        # merge filled images
        blank_temp = np.ones_like(component) * 255
        for fl in fill_img_list:
            for y in range(fl.shape[0]):
                for x in range(fl.shape[1]):
                    if fl[y][x] == 0.0:
                        blank_temp[y][x] = fl[y][x]
        # hole points
        for pt in hole_points:
            blank_temp[pt[1]][pt[0]] = 255

        return blank_temp


def autoSmoothContoursOfCharacter(grayscale):
    """
    Automatically smooth contours of grayscale
    :param grayscale:
    :return:
    """
    if grayscale is None:
        return

    # 1. Load image RGB image
    # 2. RGB image -> grayscale
    # 3. Gayscale -> bitmap
    img_gray = grayscale.copy()
    _, img_bit = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 4. Separate image to several connencted components
    components = getConnectedComponents(img_bit)
    print("components num: %d" % len(components))

    # smooth components one by one
    components_smoothed = []
    # components_smoothed.append(autoSmoothContoursOfComponent(components[1]))
    for comp in components:
        # smooth each component
        comp_smoothed = autoSmoothContoursOfComponent(comp)
        components_smoothed.append(comp_smoothed)
    for i in range(len(components)):
        cv2.imshow("cd_%d", components[1])
    for i in range(len(components_smoothed)):
        cv2.imshow("c_%d" % i, components_smoothed[i])

    # merge all smoothed components togeter to one character
    img_smoothed = createBlankGrayscaleImage(img_gray)
    for comp in components_smoothed:
        for y in range(comp.shape[0]):
            for x in range(comp.shape[1]):
                if comp[y][x] == 0.0:
                    img_smoothed[y][x] = 0.0

    return img_smoothed


def main():
    path = "test_images/src_resize.png"

    # 1. Load image
    img = cv2.imread(path)
    img_gray = None

    # 2. RGB image -> grayscale and bitmap
    if len(img.shape) == 3:
        # rgb image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif len(img.shape) == 2:
        # grayscale img
        img_gray = img

    img_smoothed = autoSmoothContoursOfCharacter(img_gray)

    cv2.imshow("img", img_gray)
    cv2.imshow("smoothed", img_smoothed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()