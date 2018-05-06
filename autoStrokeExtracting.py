# coding: utf-8
import cv2
import numpy as np

from utils.Functions import getConnectedComponents


def extractStrokesFromRadical(radical):
    """
    Stroke extracting from one radical.
    :param radical:
    :return:
    """
    if radical is None:
        return



def autoStrokeExtractiing(image):
    """
    Automatic strokes extracting
    :param image: grayscale image
    :return: strokes images with same size
    """
    if image is None:
        return

    # get connected components
    radicals = getConnectedComponents(image)
    print("radicals num: %d" % len(radicals))

    # strokes
    total_strokes = []
    for rad in radicals:
        strokes = extractStrokesFromRadical(rad)
        total_strokes += strokes

    return total_strokes


def main():
    path = "1133壬.jpg"

    img = cv2.imread(path, 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img = np.array(img, dtype=np.uint8)

    strokes = autoStrokeExtractiing(img)

    # print("storke num :%d" % len(strokes))
    #
    # for i in range(len(strokes)):
    #     cv2.imshow("stroke_%d"%i, strokes[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()