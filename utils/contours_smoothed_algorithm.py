"""
    Algorithm for contours smoothing of character.

"""
from algorithms.RDP import rdp
import cv2
import numpy as np
import math

from utils.Functions import createBlankGrayscaleImage, getAllMiniBoundingBoxesOfImage, getContourOfImage, \
                            getConnectedComponents, removeBreakPointsOfContour, sortPointsOnContourOfImage, \
                            fitCurve, draw_cubic_bezier, getSkeletonOfImage, \
                            getEndPointsOfSkeletonLine, getCrossPointsOfSkeletonLine


def autoSmoothContoursOfCharacter(grayscale, bitmap_threshold=127, eplison=10, max_error=200):
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
    _, img_bit = cv2.threshold(img_gray, bitmap_threshold, 255, cv2.THRESH_BINARY)

    # 4. Separate image to several connencted components
    components = getConnectedComponents(img_bit)
    print("components num: %d" % len(components))

    # smooth components one by one
    components_smoothed = []

    for comp in components:
        # smooth each component
        comp_smoothed = autoSmoothContoursOfComponent(comp, eplison, max_error)
        components_smoothed.append(comp_smoothed)

    # merge all smoothed components togeter to one character
    img_smoothed = createBlankGrayscaleImage(img_gray)
    for comp in components_smoothed:
        for y in range(comp.shape[0]):
            for x in range(comp.shape[1]):
                if comp[y][x] == 0.0:
                    img_smoothed[y][x] = 0.0

    return img_smoothed


def autoSmoothContoursOfComponent(component, eplison=10, max_error=200):
    """
    Automatically smooth the contours of component.
    :param component:
    :return:
    """
    if component is None:
        return

    # 5. Get contours of this component
    component_contours = getContourOfImage(component)
    contours = getConnectedComponents(component_contours, connectivity=8)
    print("contours num: ", len(contours))

    # 6. Process contours to get closed and 1-pixel width contours by removing break points
    contours_processed = []
    for cont in contours:
        cont = removeBreakPointsOfContour(cont)
        contours_processed.append(cont)
    print("contours processed num: %d" % len(contours_processed))

    # 7. Smooth contours with RDP and cubic bezeir fit curve
    contours_smoothed = []
    for cont in contours_processed:
        cont_smoothed = []
        # sorted points on contour
        cont_sorted = sortPointsOnContourOfImage(cont)

        # simplify contour with RDP
        cont_simp = rdp(cont_sorted, eplison)
        print("cont simp num: ", len(cont_simp))

        # split contour into sub-contours
        for i in range(len(cont_simp) - 1):
            start_pt = cont_simp[i]
            end_pt = cont_simp[i + 1]

            start_index = cont_sorted.index(start_pt)
            end_index = cont_sorted.index(end_pt)

            sub_cont_points = np.array(cont_sorted[start_index: end_index + 1])
            beziers = fitCurve(sub_cont_points, maxError=max_error)

            for bez in beziers:
                bezier_points = draw_cubic_bezier(bez[0], bez[1], bez[2], bez[3])
                cont_smoothed += bezier_points
        contours_smoothed.append(cont_smoothed)
    print("contours smoothed num: ", len(contours_smoothed))

    # fill black color in contour area
    if len(contours_smoothed) == 1:
        # no hole exist, directly fill black in the contour
        cont = contours_smoothed[0]
        # cont_points = sortPointsOnContourOfImage(cont)
        cont_points = np.array([cont], "int32")

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
            # cont_points = sortPointsOnContourOfImage(cont)
            cont_points = np.array([cont], "int32")

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