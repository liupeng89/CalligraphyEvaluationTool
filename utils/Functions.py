import cv2
import numpy as np
from math import sin, cos, sqrt


from skimage.measure import compare_ssim as ssim


# Resize images of soure and target, return new square images
# The width of new square should larger than the length of diagonal line of minimum bounding box.
def resizeImages(source, target):
    src_minx, src_miny, src_minw, src_minh = calculateBoundingBox(source)
    tag_minx, tag_miny, tag_minw, tag_minh = calculateBoundingBox(target)

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
    src_new_minx, src_new_miny, src_new_minw, src_new_minh = calculateBoundingBox(src_new_square)
    tag_new_minx, tag_new_miny, tag_new_minw, tag_new_minh = calculateBoundingBox(tag_new_square)

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



    # # border add extra white space:  Width * 10%
    # # source
    # new_width = src_new_square.shape[0]
    # new_height = src_new_square.shape[1]
    #
    # extra_width = int(new_width * 0.1)
    # extra_height = int(new_height * 0.1)
    #
    # new_width += extra_width
    # new_height += extra_height
    #
    # src_square = np.ones((new_width, new_height)) * 255
    # tag_square = np.ones((new_width, new_height)) * 255
    #
    # src_square[int(extra_width/2): int(extra_width/2)+src_new_square.shape[0],
    #             int(extra_height/2): int(extra_height/2) + src_new_square.shape[1]] = src_new_square
    #
    # tag_square[int(extra_width/2): int(extra_width/2)+tag_new_square.shape[0],
    #             int(extra_height/2): int(extra_height/2)+ tag_new_square.shape[1]] = tag_new_square
    #
    # ret, src_square = cv2.threshold(src_square, 127, 255, cv2.THRESH_BINARY)
    # ret, tag_square = cv2.threshold(tag_square, 127, 255, cv2.THRESH_BINARY)

    return src_square, tag_square


# Add Minimum Bounding Box to a image
def addMinBoundingBox(image):
    x, y, w, h = calculateBoundingBox(image)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return image


# Add  Minimum Bounding box to the image.
def calculateBoundingBox(image):
    if image is None:
        return None

    WIDTH = image.shape[0]
    HEIGHT = image.shape[1]

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


# Get thr rotated minimum bounding box
def getRotatedMinimumBoundingBox(image):
    if image is None:
        return None

    _, contours, _ = cv2.findContours(image, 1, 2)
    # only one object of stroke in this image
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box


# get all bounding boxes from image
def getBoundingBoxes(image):
    if image is None:
        return None
    # moments
    im2, contours, _ = cv2.findContours(image, 1, 2)

    result = []
    for i in range(len(contours)):
        item = []
        x, y, w, h = cv2.boundingRect(contours[i])
        item.append(x)
        item.append(y)
        item.append(w+2)
        item.append(h+2)

        result.append(item)

    return result



# Coverage two images: source image is red image, and target image is blue image
def coverTwoImages(source, target):

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


# Coverage image with maximum coverage rate
def shiftImageWithMaxCR(source, target):
    source = np.uint8(source)
    target = np.uint8(target)
    src_minx, src_miny, src_minw, src_minh = calculateBoundingBox(source)
    tag_minx, tag_miny, tag_minw, tag_minh = calculateBoundingBox(target)

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
            cr = calculateCR(new_tag_rect, source)
            if cr > max_cr:
                offset_x = offset_x0 + x
                offset_y = offset_y0 + y
                max_cr = cr

    new_tag_rect = np.ones(target.shape) * 255
    new_tag_rect[tag_miny + offset_y: tag_miny + offset_y + tag_minh,
    tag_minx + offset_x: tag_minx + offset_x + tag_minw] = target[tag_miny: tag_miny + tag_minh,
                                                                     tag_minx: tag_minx + tag_minw]

    return new_tag_rect


# get Center of gravity
def getCenterOfGravity(image):
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



# Coverage images with maximum overlap area
def coverageTwoImagesWithMaxOverlap(source, target):
    pass


# Coverage images with maximum SSIM
def coverageTwoImagesWithMaxSSIM(source, target):
    pass


# Calcluate Coverage Rate
def calculateCR(source, target):
    p_valid = np.sum(255.0 - source) / 255.0

    diff = target - source

    p_less = np.sum(diff == 255.0)
    p_over = np.sum(diff == -255.0)

    cr = (p_valid - p_less - p_over) / p_valid * 100.0
    return cr


# Calculate SSIM
def calculateSSIM(source, target):
    return ssim(source, target) * 100.0


# Add intersected figure of RGB image
def addIntersectedFig(image):
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


# Add squared figure of RGB image
def addSquaredFig(image):
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


# Function to konw if we have a CCW turn
def RightTurn(p1, p2, p3):
    if (p3[1]-p1[1]) * (p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
        return False
    return True


# main algorithm
def GrahamScan(P):

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


# get convex hull of image
def getConvexHullOfImage(image):
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


# Get Polygon Area (y,x)
def getPolygonArea(points):
    if points is None:
        return 0.0
    area = 0.0
    i = j = len(points)-1

    for i in range(len(points)):
        area += (points[j][1] + points[i][1]) * (points[j][0] - points[i][0])
        j = i

    return area * 0.5


# Get vaild pixels Area or number
def getValidPixelsArea(image):
    if image is None:
        return 0.0
    return np.sum(255.0 - image) / 255.0


# get the area of convex hull
def getAreaOfConvexHull(L):
    if L == None:
        return 0.0
    lines = np.hstack([L, np.roll(L, -1, axis=0)])
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
    return area


# image rotate theta degree
def rotate(image, theta):
    if image is None:
        return None

    # bounding box
    bx0, by0, bw, bh = calculateBoundingBox(image)

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


# rotate character with angle.
def rotate_character(image, angle):
    if image is None:
        return None

    image = np.uint8(image)

    # original image and four rectangle points
    rx, ry, rw, rh = calculateBoundingBox(image)
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


# return the connected components
def getConnectedComponents(image, connectivity=4):
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


# get the contours of the image
def getContourOfImage(image, minVal=100, maxVal=200):
    if image is None:
        return None
    # invert the color (background is black)
    image = 255 - image

    # use the Canny algorithm
    edge = cv2.Canny(image, minVal, maxVal)

    # invert the color of the edge image (black contour)
    edge = 255 - edge

    return edge


# get skeleton of image
def getSkeletonOfImage(image, shape=cv2.MORPH_CROSS, kernel=(3, 3)):
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



