import cv2


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

        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x + w > maxx:
            maxx = x + w
        if y + h > maxy:
            maxy = y + h
    return minx, miny, maxx-minx, maxy-miny