import cv2
import math
import numpy as np
from math import sin, cos, sqrt
from skimage.measure import compare_ssim as ssim


def resizeImages(source, target):
    """
    Resize images of source and target, in order to as much as possible to make the two images the same size.
    :param source: grayscale of source image.
    :param target: grayscale of target image.
    :return: resized-images of source and target.
    """
    src_minx, src_miny, src_minw, src_minh = getSingleMaxBoundingBoxOfImage(source)
    tag_minx, tag_miny, tag_minw, tag_minh = getSingleMaxBoundingBoxOfImage(target)

    src_min_bounding = source[src_miny: src_miny + src_minh, src_minx: src_minx + src_minw]
    tag_min_bounding = target[tag_miny: tag_miny + tag_minh, tag_minx: tag_minx + tag_minw]

    src_maxw = max(src_minw, src_minh)
    tag_maxw = max(tag_minw, tag_minh)

    src_new_square = np.ones((src_maxw, src_maxw)) * 255
    tag_new_square = np.ones((tag_maxw, tag_maxw)) * 255

    # new src square
    for y in range(src_min_bounding.shape[0]):
        for x in range(src_min_bounding.shape[1]):
            if src_min_bounding.shape[0] > src_min_bounding.shape[1]:
                # height > width
                offset = int((src_min_bounding.shape[0] - src_min_bounding.shape[1]) / 2)
                src_new_square[y][x + offset] = src_min_bounding[y][x]
            else:
                # height < width
                offset = int((src_min_bounding.shape[1] - src_min_bounding.shape[0]) / 2)
                src_new_square[y + offset][x] = src_min_bounding[y][x]

    # new tag square
    for y in range(tag_min_bounding.shape[0]):
        for x in range(tag_min_bounding.shape[1]):
            if tag_min_bounding.shape[0] > tag_min_bounding.shape[1]:
                # height > width
                offset = int((tag_min_bounding.shape[0] - tag_min_bounding.shape[1]) / 2)
                tag_new_square[y][x + offset] = tag_min_bounding[y][x]
            else:
                # height < width
                offset = int((tag_min_bounding.shape[1] - tag_min_bounding.shape[0]) / 2)
                tag_new_square[y + offset][x] = tag_min_bounding[y][x]

    # resize new square to same size between the source image and target image
    if src_new_square.shape[0] > tag_new_square.shape[0]:
        # src > tag
        src_new_square = cv2.resize(src_new_square, tag_new_square.shape)
    else:
        # src < tag
        tag_new_square = cv2.resize(tag_new_square, src_new_square.shape)

    # Border add extra white space, and the width of new square should larger than the length of diagonal
    #  line of new bounding box
    src_new_square = np.uint8(src_new_square)
    tag_new_square = np.uint8(tag_new_square)
    src_new_minx, src_new_miny, src_new_minw, src_new_minh = getSingleMaxBoundingBoxOfImage(src_new_square)
    tag_new_minx, tag_new_miny, tag_new_minw, tag_new_minh = getSingleMaxBoundingBoxOfImage(tag_new_square)

    src_diag_line = int(sqrt(src_new_minw * src_new_minw + src_new_minh * src_new_minh))
    tag_diag_line = int(sqrt(tag_new_minw * tag_new_minw + tag_new_minh * tag_new_minh))

    new_width = max(src_diag_line, tag_diag_line)

    # add
    src_square = np.ones((new_width, new_width)) * 255
    tag_square = np.ones((new_width, new_width)) * 255

    src_extra_w = new_width - src_new_square.shape[0]
    src_extra_h = new_width - src_new_square.shape[1]

    src_square[int(src_extra_w / 2): int(src_extra_w / 2) + src_new_square.shape[0],
    int(src_extra_h / 2): int(src_extra_h / 2) + src_new_square.shape[1]] = src_new_square

    tag_extra_w = new_width - tag_new_square.shape[0]
    tag_extra_h = new_width - tag_new_square.shape[1]

    tag_square[int(tag_extra_w / 2): int(tag_extra_w / 2) + tag_new_square.shape[0],
    int(tag_extra_h / 2): int(tag_extra_h / 2) + tag_new_square.shape[1]] = tag_new_square

    ret, src_square = cv2.threshold(src_square, 127, 255, cv2.THRESH_BINARY)
    ret, tag_square = cv2.threshold(tag_square, 127, 255, cv2.THRESH_BINARY)

    return src_square, tag_square


def addMinBoundingBox(image):
    """
    Adding the minimizing bounding rectangle boxing with green color to the RGB image of character.
    :param image: Grayscale image of character.
    :return: RGB image of character with green minimizing bounding boxing.
    """
    x, y, w, h = getSingleMaxBoundingBoxOfImage(image)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return image


def getSingleMaxBoundingBoxOfImage(image):
    """
    Calculate the coordinates(x, y, w, h) of single maximizing bounding rectangle boxing of grayscale image
    of character, in order to using this bounding box to select the region of character.
    :param image: grayscale image of character.
    :return: coordinates(x, y, w, h) of single maximizing bounding boxing.
    """
    if image is None:
        return None

    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]

    # moments
    im2, contours, hierarchy = cv2.findContours(image, 1, 2)

    minx = WIDTH
    miny = HEIGHT
    maxx = 0
    maxy = 0
    # Bounding box
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        if w > 0.95 * WIDTH and h > 0.95 * HEIGHT:
            continue
        minx = min(x, minx); miny = min(y, miny)
        maxx = max(x+w, maxx); maxy = max(y+h, maxy)

    return minx, miny, maxx-minx, maxy-miny


def getAllMiniBoundingBoxesOfImage(image):
    """
    Get all minimizing bounding boxes in the grayscale image of character. In order to select independented no-connected
    region of character.
    :param image: grayscale image of character.
    :return: list of bounding boxes of character.
    """
    boxes = []
    if image is None:
        return boxes

    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]

    # moments
    im2, contours, hierarchy = cv2.findContours(image, 1, 2)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        if w > 0.95 * WIDTH and h > 0.95 * HEIGHT:
            continue
        boxes.append((x, y, w, h))

    return boxes


def getRotatedMinimumBoundingBox(image):
    """
    Get the rotated minimizing bounding boxes of grayscale image of character.
    :param image: grayscale image of character.
    :return: list of rotated minimizing bounding boxes.
    """
    if image is None:
        return None

    _, contours, _ = cv2.findContours(image, 1, 2)
    # only one object of stroke in this image
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box


def coverTwoImages(source, target):
    """
    Using the target image (blue color) to cover the source image (red color).
    :param source: grayscale image of source.
    :param target: grayscale image of target.
    :return: RGB image with using the target image (blue color) to cover the source image (red color).
    """

    # grayscale images to RGB images
    WIDTH, HEIGHT = source.shape

    coverage_img = np.ones((WIDTH, HEIGHT, 3)) * 255

    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            if source[y][x] == 0.0 and target[y][x] == 255.0:
                # red
                coverage_img[y][x] = (0, 0, 255)
            elif source[y][x] == 0.0 and target[y][x] == 0.0:
                # mix color overlap area : black
                coverage_img[y][x] = (0, 0, 0)
            elif source[y][x] == 255.0 and target[y][x] == 0.0:
                # blue
                coverage_img[y][x] = (255, 0, 0)
            else:
                # blue
                coverage_img[y][x] = (255, 255, 255)

    return coverage_img


def shiftImageWithMaxCR(source, target):
    """
    Shift the target image based on the maximizing coverage rate with the source image.
    :param source: grayscale image of source.
    :param target: grayscale image of target.
    :return: Shifted target image.
    """
    source = np.uint8(source)
    target = np.uint8(target)
    src_minx, src_miny, src_minw, src_minh = getSingleMaxBoundingBoxOfImage(source)
    tag_minx, tag_miny, tag_minw, tag_minh = getSingleMaxBoundingBoxOfImage(target)

    # new rect of src and tag images
    new_rect_x = min(src_minx, tag_minx)
    new_rect_y = min(src_miny, tag_miny)
    new_rect_w = max(src_minx+src_minw, tag_minx+tag_minw) - new_rect_x
    new_rect_h = max(src_miny+src_minh, tag_miny+tag_minh) - new_rect_y

    # offset 0
    offset_y0 = -tag_miny
    offset_x0 = -tag_minx

    # print("Offset o: (%d, %d)" % (offset_x0, offset_y0))

    diff_x = source.shape[0] - tag_minw
    diff_y = source.shape[1] - tag_minh

    offset_x = 0
    offset_y = 0

    max_cr = -1000.0
    for y in range(diff_y):
        for x in range(diff_x):
            new_tag_rect = np.ones(target.shape) * 255
            new_tag_rect[tag_miny + offset_y0 + y: tag_miny + offset_y0 + y + tag_minh,
                    tag_minx + offset_x0 + x: tag_minx + offset_x0 + x + tag_minw] = target[tag_miny: tag_miny + tag_minh,
                                                                 tag_minx: tag_minx + tag_minw]
            cr = calculateCoverageRate(new_tag_rect, source)
            if cr > max_cr:
                offset_x = offset_x0 + x
                offset_y = offset_y0 + y
                max_cr = cr

    new_tag_rect = np.ones(target.shape) * 255
    new_tag_rect[tag_miny + offset_y: tag_miny + offset_y + tag_minh,
    tag_minx + offset_x: tag_minx + offset_x + tag_minw] = target[tag_miny: tag_miny + tag_minh,
                                                                     tag_minx: tag_minx + tag_minw]

    return new_tag_rect


def getCenterOfGravity(image):
    """
    Get the center of gravity of image.
    :param image: grayscale image of character.
    :return: (x, y), the coordinate of center of gravity of image.
    """
    src_cog_x = 0; src_cog_y = 0
    total_pixels = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] == 0.0:
                src_cog_x += x
                src_cog_y += y
                total_pixels += 1

    src_cog_x = int(src_cog_x / total_pixels)
    src_cog_y = int(src_cog_y / total_pixels)
    return src_cog_x, src_cog_y


def calculateCoverageRate(source, target):
    """
    Corverage rate calculation.
    :param source: grayscale image of source.
    :param target: grayscale image of target.
    :return: Coverage rate of source and target images.
    """
    p_valid = np.sum(255.0 - source) / 255.0

    if p_valid == 0.0:
        return 0.0

    diff = target - source

    p_less = np.sum(diff == 255.0)
    p_over = np.sum(diff == -255.0)

    cr = (p_valid - p_less - p_over) / p_valid * 100.0
    return cr


def calculateSSIM(source, target):
    """
    SSIM calculation.
    :param source: grayscale image of source.
    :param target: grayscale image of target.
    :return: SSIM of source and target images.
    """
    return ssim(source, target) * 100.0


def addIntersectedFig(image):
    """
    Adding the intersected lines in the RGB image of character.
    :param image: grayscale image of character.
    :return: RGB image of character with intersected lines.
    """
    if image is None:
        return None

    bk_img = np.ones(image.shape) * 255

    # bk add border lines, width=3, color = red
    h, w, _ = bk_img.shape
    bk_img = cv2.line(bk_img, (2, 2), (h-4, 2), (0, 0, 255), 5)
    bk_img = cv2.line(bk_img, (2, 2), (2, w - 4), (0, 0, 255), 5)
    bk_img = cv2.line(bk_img, (h - 4, 2), (h - 4, w - 4), (0, 0, 255), 5)
    bk_img = cv2.line(bk_img, (2, w - 4), (h - 4, w - 4), (0, 0, 255), 5)

    # middle lines
    middle_x = int(w/2)
    middle_y = int(h/2)
    bk_img = cv2.line(bk_img, (middle_y, 0), (middle_y, w-1), (0, 255, 0), 2)
    bk_img = cv2.line(bk_img, (0, middle_x), (h-1, middle_x), (0, 255, 0), 2)

    # cross lines
    bk_img = cv2.line(bk_img, (0, 0), (h-1, w-1), (0, 255, 0), 2)
    bk_img = cv2.line(bk_img, (h-1, 0), (0, w-1), (0, 255, 0), 2)

    # crop bk and image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x][0] != 255 or image[y][x][1] != 255 or image[y][x][2] != 255:
                bk_img[y][x] = image[y][x]

    return bk_img


def addSquaredFig(image):
    """
    Adding the squared lines in the RGB image of character.
    :param image: grayscale image of character.
    :return: the RGB image of character with squared lines.
    """
    if image is None:
        return None

    bk_img = np.ones(image.shape) * 255

    # bk add border lines, width=3, color = red
    h, w, _ = bk_img.shape
    bk_img = cv2.line(bk_img, (2, 2), (h-4, 2), (0, 0, 255), 5)
    bk_img = cv2.line(bk_img, (2, 2), (2, w - 4), (0, 0, 255), 5)
    bk_img = cv2.line(bk_img, (h - 4, 2), (h - 4, w - 4), (0, 0, 255), 5)
    bk_img = cv2.line(bk_img, (2, w - 4), (h - 4, w - 4), (0, 0, 255), 5)

    # 1/3w, 2/3w, 1/3h, 2/3h
    w13 = int(w / 3)
    w23 = int(2 * w / 3)
    h13 = int(h / 3)
    h23 = int(2 * h / 3)

    # lines
    bk_img = cv2.line(bk_img, (h13, 0), (h13, w - 1), (0, 0, 255), 2)
    bk_img = cv2.line(bk_img, (h23, 0), (h23, w - 1), (0, 0, 255), 2)

    bk_img = cv2.line(bk_img, (0, w13), (h - 1, w13), (0, 0, 255), 2)
    bk_img = cv2.line(bk_img, (0, w23), (h - 1, w23), (0, 0, 255), 2)

    # crop bk and image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x][0] != 255 or image[y][x][1] != 255 or image[y][x][2] != 255:
                bk_img[y][x] = image[y][x]

    return bk_img


def RightTurn(p1, p2, p3):
    """
    Determing whether three points constitute a "left-turn" or "right-turn" by computing the z-coordinate of the
    cross product of the two vectors P1P2 and P2P3, which is given by the expression (x2-x1)(y3-y1)-(y2-y1)(x3-x1). If
    the result is 0, the points are colinear; if it is positive, the three points constitute a "left-turn" or counter-
    clockwise (CCW) orientation, otherwise a "right-turn" or clockwise orientation.
    :param p1: (x,y) of point P1
    :param p2: (x,y) of point P2
    :param p3: (x,y) of point P3
    :return: is "right-turn" or not.
    """
    if (p3[1]-p1[1]) * (p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
        return False
    return True


def GrahamScan(P):
    """
    Graham Scan algorithm implementation of finding the convex hull of a finite set of points
    :param P: list of coordinates of points.
    :return: list of points on the convex hull boundary.
    """

    P.sort()
    L_upper = [P[0], P[1]]
    # Compute the upper part of the hull
    for i in range(2, len(P)):
        L_upper.append(P[i])
        while len(L_upper) > 2 and not RightTurn(L_upper[-1], L_upper[-2], L_upper[-3]):
            del L_upper[-2]
    L_lower = [P[-1], P[-2]]
    # compute the lower part of the hull
    for i in range(len(P)-3, -1, -1):
        L_lower.append(P[i])
        while len(L_lower) > 2 and not RightTurn(L_lower[-1], L_lower[-2], L_lower[-3]):
            del L_lower[-2]

    del L_lower[0]
    del L_lower[-1]
    L = L_upper + L_lower
    return np.array(L)


def getConvexHullOfImage(image):
    """
    Get the convex hull of grayscale image of character.
    :param image: graysale image of character.
    :return: list of points on convex hull boundary of character.
    """
    if image is None:
        return None

    # P and L
    P = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] == 0.0:
                P.append((x, y))

    # Graham Scan algorithm
    L = GrahamScan(P)
    return L


def calculatePolygonArea(points):
    """
    Calculate the area of polygon.
    :param points: the end points of the polygon.
    :return: the area of polygon
    """
    if points is None:
        return 0.0
    area = 0.0
    i = j = len(points)-1

    for i in range(len(points)):
        area += (points[j][1] + points[i][1]) * (points[j][0] - points[i][0])
        j = i

    return area * 0.5


def calculateValidPixelsArea(image):
    """
    Calculate the area of valid pixels region of grayscale image of character.
    :param image: grayscale image of character.
    :return: the area of valid pixels region of character.
    """
    if image is None:
        return 0.0
    return np.sum(255.0 - image) / 255.0


def calculateConvexHullArea(L):
    """
    Calculate the area of convex hull of grayscale image of character.
    :param L: the end points of convex hull.
    :return: the area of convex hull region.
    """
    if L == None:
        return 0.0
    lines = np.hstack([L, np.roll(L, -1, axis=0)])
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
    return area


def rotateImage(image, theta):
    """
    Rotate image with theta degree.
    :param image: grayscale image of character.
    :param theta: rotation angle degree.
    :return: the rotated grayscale image of character.
    """
    if image is None:
        return None

    # bounding box
    bx0, by0, bw, bh = getSingleMaxBoundingBoxOfImage(image)

    # center of retota
    x0 = bx0 + int(bw/2)
    y0 = by0 + int(bh/2)

    # new image of square
    new_img = np.ones(image.shape) * 255

    # rotate
    for y in range(by0, by0+bh):
        for x in range(bx0, bx0+bw):
            x2 = round(cos(theta) * (x-x0) - sin(theta) * (y-y0)) + x0
            y2 = round(sin(theta) * (x-x0) + cos(theta) * (y-y0)) + y0

            new_img[y2][x2] = image[y][x]

    return new_img


def rotate_character(image, angle):
    """
    Rotate grayscale image of character.
    :param image: grayscale image of character.
    :param angle: rotation angle degree.
    :return: rotated grayscale image of character.
    """
    if image is None:
        return None

    image = np.uint8(image)

    # original image and four rectangle points
    rx, ry, rw, rh = getSingleMaxBoundingBoxOfImage(image)
    cx = rx + int(rw/2)
    cy = ry + int(rh/2)

    # invert color from white to black background
    image = 255 - image

    # rotate
    M = cv2.getRotationMatrix2D((cy, cx), angle, 1)
    dst = cv2.warpAffine(image, M, image.shape)

    # invert color from black to white background
    dst = 255 - dst
    dst = np.uint8(dst)
    return dst


def splitConnectedComponents(image, connectivity=8):
    """
    Extract the connected components of character from image with Labeling algorithm.
    :param image: image of grayscale image.
    :param connectivity: connectivity of image (4, or 8)
    :return: independented components of characters.
    """
    image_ = 255 - image

    ret, labels = cv2.connectedComponents(image_, connectivity=connectivity)
    components = []
    for r in range(1, ret):
        img_ = np.ones(image_.shape, dtype=np.uint8) * 255
        for y in range(image_.shape[0]):
            for x in range(image_.shape[1]):
                if labels[y][x] == r:
                    img_[y][x] = 0.0
        components.append(img_)

    return components


def getConnectedComponents(image, connectivity=4):
    """
    Get the connected components of character from image with Labeling algorithm.
    :param image: image of grayscale image.
    :param connectivity: connectivity of image (4, or 8)
    :return: independented components of characters.
    """
    if image is None:
        return None
    # the image is black vaild pixels and white background pixels
    image = cv2.bitwise_not(image) # inverting the color
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity)
    components = []
    if num_labels == 0:
        return None

    for i in range(1, num_labels):
        mask = labels==i
        mask = np.uint8(mask)
        fig = cv2.bitwise_and(image, image, mask=mask)
        fig = cv2.bitwise_not(fig)
        components.append(fig)

    return components


def getContourOfImage(image, minVal=100, maxVal=200):
    """
    Get the contour of grayscale image of character by using the Canny Edge Detection algorithm.
    :param image: grayscale image of character.
    :param minVal: minimizing value of Hysteresis thresholding.
    :param maxVal: maximizing value of Hysteresis thresholding.
    :return: grayscale image of edge.
    """
    if image is None:
        return None
    # invert the color (background is black)
    image = 255 - image

    # use the Canny algorithm
    edge = cv2.Canny(image, minVal, maxVal)

    # invert the color of the edge image (black contour)
    edge = 255 - edge

    return edge


def removeBreakPointsOfContour(contour):
    """
    Remove the break points of contour to keep the contour close.
    :param contour:
    :return:
    """
    if contour is None:
        return
    # find all break points
    break_points = []
    for y in range(1, contour.shape[0]-1):
        for x in range(1, contour.shape[1]-1):
            if contour[y][x] == 0.0:

                num_ = getNumberOfValidPixels(contour, x, y)
                if num_ == 1:
                    # break points
                    break_points.append((x, y))
    print("number of break points: %d" % len(break_points))
    if len(break_points) % 2 != 0:
        print("break points should be even number!")
    bp_label = []
    for id in range(len(break_points)):
        bp_label.append(0.0)

    # remove the break points pair
    for id in range(len(break_points)):
        if bp_label[id] == 1.:
            continue
        if bp_label[id] == 0.0:
            bp_label[id] = 1.
        max_dist = 100000.
        start_point = break_points[id]
        next_point = None
        next_id = 0
        for idx in range(id+1, len(break_points)):
            if bp_label[idx] == 1.:
                continue
            dist_ = math.sqrt((start_point[0] - break_points[idx][0])**2 + (start_point[1] - break_points[idx][1])**2)
            if dist_ < max_dist:
                max_dist = dist_
                next_point = (break_points[idx][0], break_points[idx][1])
                next_id = idx
        if next_point is None:
            continue
        else:
            contour = cv2.line(contour, start_point, next_point, color=0, thickness=1)
            bp_label[next_id] = 1.

    return contour





def getSkeletonOfImage(image, shape=cv2.MORPH_CROSS, kernel=(3, 3)):
    """
    Get the skeletion of grayscale image of character.
    :param image: grayscale image of character.
    :param shape: element shape that could be one of the following: MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS, and
        CV_SHAPE_CUSTOM.
    :param kernel: size of the structuring element.
    :return: the skeleton grayscale image of character.
    """
    if image is None:
        return None

    # original image invert color
    image = 255 - image

    # skeleton
    skel = np.zeros(image.shape, np.uint8)
    size = np.size(image)

    element = cv2.getStructuringElement(shape, kernel)
    done = False

    while (not done):
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    # invert the color of skeleton image
    skel = 255 - skel
    return skel


def getNumberOfValidPixels(image, x, y):
    """
    Get the number of valid pixels of the 8-neighbours of (x, y)
    :param image: grayscale image of character
    :param x: x value of Point (x, y)
    :param y: y value of Point (x, y)
    :return: the number of valid pixels of the 8-neighbours of (x, y)
    """
    valid_num = 0

    # X2 point
    if image[y - 1][x] == 0.0:
        valid_num += 1

    # X3 point
    if image[y - 1][x + 1] == 0.0:
        valid_num += 1

    # X4 point
    if image[y][x + 1] == 0.0:
        valid_num += 1

    # X5 point
    if image[y + 1][x + 1] == 0.0:
        valid_num += 1

    # X6 point
    if image[y + 1][x] == 0.0:
        valid_num += 1

    # X7 point
    if image[y + 1][x - 1] == 0.0:
        valid_num += 1

    # X8 point
    if image[y][x - 1] == 0.0:
        valid_num += 1

    # X9 point
    if image[y - 1][x - 1] == 0.0:
        valid_num += 1

    return valid_num


def getEndPointsOfSkeletonLine(image):
    """
    Obtain the end points of skeleton line, suppose the image is the skeleton image(white background and black
    skeleton line).
    :param image: the skeleton grayscale image of character
    :return: the end points of skeleton line
    """
    end_points = []
    if image is None:
        return end_points

    # find the end points which number == 1
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y][x] == 0.0:
                # black points
                black_num = getNumberOfValidPixels(image, x, y)

                # end points
                if black_num == 1:
                    end_points.append((x, y))
    return end_points


def getCrossAreaPointsOfSkeletionLine(image):
    """
    Get all cross points in the cross area of skeleton lines of character.
    :param image: the skeleton grayscale image of character.
    :return: the cross points in the cross area of the skeleton lines of character.
    """
    cross_points = []

    if image is None:
        return cross_points
    # find cross points
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y][x] == 0.0:
                # black points
                black_num = getNumberOfValidPixels(image, x, y)

                # cross points
                if black_num >= 3:
                    cross_points.append((x, y))
    print("cross points len : %d" % len(cross_points))
    return cross_points


def getCrossPointsOfSkeletonLine(image):
    """
    Get the cross points of skeleton line.
    :param image: the skeleton grayscale image of character.
    :return: cross points of skeleton line of character.
    """
    cross_points = []
    cross_points_no_extra = []
    if image is None:
        return cross_points
    # find cross points
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y][x] == 0.0:
                # black points
                black_num = getNumberOfValidPixels(image, x, y)

                # cross points
                if black_num >= 3:
                    cross_points.append((x, y))
    print("cross points len : %d" % len(cross_points))

    # remove the extra cross points and maintain the single cross point of several close points
    for (x, y) in cross_points:
        black_num = 0
        # P2
        if image[y-1][x] == 0.0 and (x, y-1) in cross_points:
            black_num += 1
        # P4
        if image[y][x+1] == 0.0 and (x+1, y) in cross_points:
            black_num += 1
        # P6
        if image[y+1][x] == 0.0 and (x, y+1) in cross_points:
            black_num += 1
        # P8
        if image[y][x-1] == 0.0 and (x-1, y) in cross_points:
            black_num += 1

        if black_num == 2 or black_num == 3 or black_num == 4:
            print(black_num)
            cross_points_no_extra.append((x, y))

        if (x, y) in cross_points and (x, y-1) not in cross_points and (x+1, y-1) not in cross_points and (x+1, y) not in \
            cross_points and (x+1, y+1) not in cross_points and (x, y+1) not in cross_points and (x-1, y+1) not in \
            cross_points and (x-1, y) not in cross_points and (x-1, y-1) not in cross_points:
            cross_points_no_extra.append((x, y))

    return cross_points_no_extra


def removeBranchOfSkeletonLine(image, end_points, cross_points, DIST_THRESHOLD=20):
    """
    Remove the extra branches of skeleton lines of character.
    :param DIST_THRESHOLD:
    :param image: the skeleton grayscale image of character.
    :param end_points: all end points of the skeleton lines.
    :param cross_points: all cross points of the skeleton lines.
    :param DIST_THRESHOLD: the distance threshold from cross points to end points.
    :return: skeleton grayscale image of character without extra branches.
    """

    if image is None:
        return None
    # image = image.copy()
    # remove branches of skeleton line
    for (c_x, c_y) in cross_points:
        for (e_x, e_y) in end_points:
            dist = math.sqrt((c_x-e_x) * (c_x-e_x) + (c_y-e_y) * (c_y-e_y))
            if dist < DIST_THRESHOLD:
                print("%d %d %d %d " % (e_x, e_y, c_x, c_y))
                branch_points = getPointsOfExtraBranchOfSkeletonLine(image, e_x, e_y, c_x, c_y)
                print("branch length: %d" % len(branch_points))

                # remove branch points
                for (bx, by) in branch_points:
                    image[by][bx] = 255

    return image


def getPointsOfExtraBranchOfSkeletonLine(image, start_x, start_y, end_x, end_y):
    """
    Get all points in the extra branch of skeleton lines from cross points to end points.
    :param image: skeleton grayscale image of character.
    :param start_x: x-axis value of cross point.
    :param start_y: y-axis value of cross point.
    :param end_x: x-axis value of end point.
    :param end_y: y-axis value of end point.
    :return: all points in the extra branch.
    """
    extra_branch_points = []
    start_point = (start_x, start_y)
    next_point = start_point

    while(True):

        # P2
        if image[start_point[1]-1][start_point[0]] == 0.0 and (start_point[0], start_point[1]-1) not in \
                extra_branch_points:
            next_point = (start_point[0], start_point[1]-1)
            print(next_point)
        # P3
        if image[start_point[1]-1][start_point[0]+1] == 0.0 and (start_point[0]+1, start_point[1]-1) not in \
                extra_branch_points:
            next_point = (start_point[0]+1, start_point[1]-1)
            print(next_point)
        # P4
        if image[start_point[1]][start_point[0]+1] == 0.0 and (start_point[0]+1, start_point[1]) not in \
                extra_branch_points:
            next_point = (start_point[0]+1, start_point[1])
            print(next_point)
        # P5
        if image[start_point[1]+1][start_point[0]+1] == 0.0 and (start_point[0]+1, start_point[1]+1) not in \
                extra_branch_points:
            next_point = (start_point[0]+1, start_point[1]+1)
            print(next_point)
        # P6
        if image[start_point[1]+1][start_point[0]] == 0.0 and (start_point[0], start_point[1]+1) not in \
                extra_branch_points:
            next_point = (start_point[0], start_point[1]+1)
            print(next_point)
        # P7
        if image[start_point[1]+1][start_point[0]-1] == 0.0 and (start_point[0]-1, start_point[1]+1) not in \
                extra_branch_points:
            next_point = (start_point[0]-1, start_point[1]+1)
            print(next_point)
        # P8
        if image[start_point[1]][start_point[0]-1] == 0.0 and (start_point[0]-1, start_point[1]) not in \
                extra_branch_points:
            next_point = (start_point[0]-1, start_point[1])
            print(next_point)
        # P9
        if image[start_point[1]-1][start_point[0]-1] == 0.0 and (start_point[0]-1, start_point[1]-1) not in \
                extra_branch_points:
            next_point = (start_point[0]-1, start_point[1]-1)
            print(next_point)

        extra_branch_points.append(start_point)

        if next_point[0] == end_x and next_point[1] == end_y:
            # next point is the cross point
            break
        else:
            start_point = next_point

    return extra_branch_points


def sortPointsOnContourOfImage(image, isClockwise=True):
    """
    Sort the points on contour with clockwise direction or counter-clockwise direction
    :param image: contour grayscale image of character.
    :param isClockwise: clockwise or counter-clockwise.
    :return: list of sorted points on contour of character.
    """
    if image is None:
        return
    contour_points = []
    # find the begin point
    start_point = None
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] == 0.0:
                # first black point is the start point
                start_point = (x, y)
                break
        if start_point:
            break

    print("begin point: " + str(start_point))

    # find second points with different direction of clockwise and counter-clockwise
    second_point = None
    # clockwise direction
    if image[y][x + 1] == 0.0:
        # P4 position
        second_point = (x + 1, y)
    elif image[y + 1][x + 1] == 0.0:
        # P5 positon
        second_point = (x + 1, y + 1)

    # contour points
    contour_points.append(start_point)
    contour_points.append(second_point)

    # contour point lables
    contour_order_lables = np.zeros_like(image)
    contour_order_lables[start_point[1]][start_point[0]] = 1.
    contour_order_lables[second_point[1]][second_point[0]] = 1.

    next_point = second_point
    current_point = second_point
    while True:
        x = current_point[0]; y = current_point[1]

        # 2,4,6,8 position firstly and then 3,5,7,9 position
        # point in 2 position
        if image[y - 1][x] == 0.0 and contour_order_lables[y - 1][x] == 0.0:
            next_point = (x, y - 1)
            contour_order_lables[y - 1][x] = 1.

        # point in 4 position
        elif image[y][x + 1] == 0.0 and contour_order_lables[y][x + 1] == 0.0:
            next_point = (x + 1, y)
            contour_order_lables[y][x + 1] = 1.

        # point in 6 position
        elif image[y + 1][x] == 0.0 and contour_order_lables[y + 1][x] == 0.0:
            next_point = (x, y + 1)
            contour_order_lables[y + 1][x] = 1.

        # point in 8 position
        elif image[y][x - 1] == 0.0 and contour_order_lables[y][x - 1] == 0.0:
            next_point = (x - 1, y)
            contour_order_lables[y][x - 1] = 1.

        # point in 3 position
        elif image[y - 1][x + 1] == 0.0 and contour_order_lables[y - 1][x + 1] == 0.0:
            next_point = (x + 1, y - 1)
            contour_order_lables[y - 1][x + 1] = 1.

        # point in 5 position
        elif image[y + 1][x + 1] == 0.0 and contour_order_lables[y + 1][x + 1] == 0.0:
            next_point = (x + 1, y + 1)
            contour_order_lables[y + 1][x + 1] = 1.

        # point in 7 position
        elif image[y + 1][x - 1] == 0.0 and contour_order_lables[y + 1][x - 1] == 0.0:
            next_point = (x - 1, y + 1)
            contour_order_lables[y + 1][x - 1] = 1.

        # point in 9 position
        elif image[y - 1][x - 1] == 0.0 and contour_order_lables[y - 1][x - 1] == 0.0:
            next_point = (x - 1, y - 1)
            contour_order_lables[y - 1][x - 1] = 1.

        if next_point == current_point:
            contour_points.append(current_point)
            break
        else:
            contour_points.append(current_point)
            current_point = next_point

    if isClockwise:
        return contour_points
    else:
        contour_points.reverse()
        return contour_points


def cubic_bezier_sum(t, w):
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    return w[0] * mt3 + 3 * w[1] * mt2 * t + 3 * w[2] * mt * t2 + w[3] * t3


def draw_cubic_bezier(p1, p2, p3, p4):
    points = []
    t = 0
    while t < 1:
        x = int(cubic_bezier_sum(t, (p1[0], p2[0], p3[0], p4[0])))
        y = int(cubic_bezier_sum(t, (p1[1], p2[1], p3[1], p4[1])))

        points.append((x, y))

        t += 0.01
    return points


def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTanget = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTanget, maxError)


def fitCubic(points, leftTangent, rightTangent, error):
    # Use heuristic if region only has two points in it
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.
        bezCurve = [points[0], points[0]+leftTangent*dist, points[1]+rightTangent*dist, points[1]]
        return [bezCurve]

    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return [bezCurve]

    # If error not too large, try some reparameterization and iteration
    if maxError < error**2:
        for i in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint-1] - points[splitPoint+1])
    beziers += fitCubic(points[:splitPoint+1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers


def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    # compute the A's
    A = np.zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = leftTangent * 3 * (1-u)**2 *u
        A[i][1] = rightTangent * 3 * (1-u) * u**2

    # Create the C and X matrics
    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] += np.dot(A[i][0], A[i][1])
        C[1][1] += np.dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)
        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    # compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = np.linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezCurve[1] = bezCurve[0] + leftTangent * (segLength / 3.)
        bezCurve[2] = bezCurve[3] + rightTangent * (segLength / 3.)
    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bezCurve[1] = bezCurve[0] + leftTangent * alpha_l
        bezCurve[2] = bezCurve[3] + rightTangent * alpha_r

    return bezCurve


def reparameterize(bezier, points, parameters):
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters)]


def newtonRaphsonRootFind(bez, point, u):
    """
        Newton's root finding algorithm calculates f(x)=0 by reiterating
       x_n+1 = x_n - f(x_n)/f'(x_n)
       We are trying to find curve parameter u for some point p that minimizes
       the distance from that point to the curve. Distance point to curve is d=q(u)-p.
       At minimum distance the point is perpendicular to the curve.
       We are solving
       f = q(u)-p * q'(u) = 0
       with
       f' = q'(u) * q'(u) + q(u)-p * q''(u)
       gives
       u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    :param bez:
    :param point:
    :param u:
    :return:
    """
    d = q(bez, u) - point
    numerator = (d * qprime(bez, u)).sum()
    denominator = (qprime(bez, u)**2 + d * qprimeprime(bez, u)).sum()
    if denominator == 0.0:
        return u
    else:
        return u - numerator / denominator

def chordLengthParameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i-1] + np.linalg.norm(points[i] - points[i-1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]
    return u

def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points) / 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = np.linalg.norm(q(bez, u)-point)**2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i

    return maxDist, splitPoint


def normalize(v):
    return v / np.linalg.norm(v)


# evaluates cubic bezier at t, return point
def q(ctrlPoly, t):
    return (1.0-t)**3 * ctrlPoly[0] + 3*(1.0-t)**2 * t * ctrlPoly[1] + 3*(1.0-t)* t**2 * ctrlPoly[2] + t**3 * ctrlPoly[3]


# evaluates cubic bezier first derivative at t, return point
def qprime(ctrlPoly, t):
    return 3*(1.0-t)**2 * (ctrlPoly[1]-ctrlPoly[0]) + 6*(1.0-t) * t * (ctrlPoly[2]-ctrlPoly[1]) + 3*t**2 * (ctrlPoly[3]-ctrlPoly[2])


# evaluates cubic bezier second derivative at t, return point
def qprimeprime(ctrlPoly, t):
    return 6*(1.0-t) * (ctrlPoly[2]-2*ctrlPoly[1]+ctrlPoly[0]) + 6*(t) * (ctrlPoly[3]-2*ctrlPoly[2]+ctrlPoly[1])


def getCenterOfRectangles(rect):
    """
    Get the coordinate (x, y) of the center if rectangle.
    :param rect: rectangle (x, y, w, h)
    :return: (x, y) of center of rectangle.
    """
    if rect is None:
        return None
    cx = rect[0] + int(rect[2]/2)
    cy = rect[1] + int(rect[3]/2)

    return (cx, cy)


def combineRectangles(rectangles, rect_list):
    """
    Combining rectangles together.
    :param rectangles: list of rectangles.
    :param rect_list: list of index of rectangles in rectangles list.
    :return: new combined rectangle.
    """
    if rectangles is None:
        return
    if len(rect_list) == 1:
        return rectangles[rect_list[0]]

    new_rect_x0 = rectangles[rect_list[0]][0]
    new_rect_y0 = rectangles[rect_list[0]][1]
    new_rect_x1 = new_rect_x0 + rectangles[rect_list[0]][2]
    new_rect_y1 = new_rect_y0 + rectangles[rect_list[0]][3]


    for id in range(1, len(rect_list)):
        print(id)
        rect_x0 = rectangles[rect_list[id]][0]
        rect_y0 = rectangles[rect_list[id]][1]
        rect_x1 = rect_x0 + rectangles[rect_list[id]][2]
        rect_y1 = rect_y0 + rectangles[rect_list[id]][3]

        new_rect_x0 = min(new_rect_x0, rect_x0)
        new_rect_y0 = min(new_rect_y0, rect_y0)

        new_rect_x1 = max(new_rect_x1, rect_x1)
        new_rect_y1 = max(new_rect_y1, rect_y1)

    return new_rect_x0, new_rect_y0, new_rect_x1-new_rect_x0, new_rect_y1-new_rect_y0


