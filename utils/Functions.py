import cv2
import numpy as np

from skimage.measure import compare_ssim as ssim


# Resize images of soure and target, return new square images
def resizeImages(source, target):
    src_minx, src_miny, src_minw, src_minh = addBoundingBox(source)
    tag_minx, tag_miny, tag_minw, tag_minh = addBoundingBox(target)

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

    # border add extra white space:  Width * 10%
    # source
    new_width = src_new_square.shape[0]
    new_height = src_new_square.shape[1]

    extra_width = int(new_width * 0.1)
    extra_height = int(new_height * 0.1)

    new_width += extra_width
    new_height += extra_height

    src_square = np.ones((new_width, new_height)) * 255
    tag_square = np.ones((new_width, new_height)) * 255

    src_square[int(extra_width/2): int(extra_width/2)+src_new_square.shape[0],
                int(extra_height/2): int(extra_height/2) + src_new_square.shape[1]] = src_new_square

    tag_square[int(extra_width/2): int(extra_width/2)+tag_new_square.shape[0],
                int(extra_height/2): int(extra_height/2)+ tag_new_square.shape[1]] = tag_new_square

    return src_square, tag_square


# Add  Minimum Bounding box to the image.
def addBoundingBox(image):
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

    return coverage_img


# Coverage image with maximum coverage rate
def coverageTwoImagesWithMaxCR(source, target):
    src_minx, src_miny, src_minw, src_minh = addBoundingBox(source)
    tag_minx, tag_miny, tag_minw, tag_minh = addBoundingBox(target)

    new_rect = [min(src_minx, tag_minx), min(src_miny, tag_miny), max(src_minw, tag_minw), max(src_minh, tag_minh)]

    src_rect = source[src_miny: src_miny+src_minh, src_minx: src_minx+src_minw]
    tag_rect = target[tag_miny: tag_miny+tag_minh, tag_minx: tag_minx+tag_minw]


# Coverage images with maximum overlap area
def coverageTwoImagesWithMaxOverlap(source, target):
    pass


# Coverage images with maximum SSIM
def coverageTwoImagesWithMaxSSIM(source, target):
    pass


# Calcluate Coverage Rate
def calculateCR(source, target):
    p_valid = np.sum(255.0 - source) / 255.0

    p_less = 0; p_over = 0

    diff = target - source
    for y in range(diff.shape[0]):
        for x in range(diff.shape[1]):
            if diff[y][x] == 255.0:
                # less
                p_less += 1
            elif diff[y][x] == -255.0:
                # over
                p_over += 1
    cr = (p_valid - p_less - p_over) / p_valid * 100.0
    return cr


# Calculate SSIM
def calculateSSIM(source, target):
    return ssim(source, target) * 100.0


# Add intersected figure of RGB image
def addIntersectedFig(image):
    pass


# Add squared figure of RGB image
def addSquaredFig(image):
    pass