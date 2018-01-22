import numpy as np
import cv2


def main():
    src_path = "../strokes/src_resize 7.png"
    tag_path = "../strokes/tag_resize 7.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)
    src_img_ = 255 - src_img
    tag_img_ = 255 - tag_img

    src_edges = cv2.Canny(src_img, 100, 200)

    tag_edges = cv2.Canny(tag_img, 100, 200)


    # skeleton
    src_skel = np.zeros(src_img.shape, np.uint8)
    size = np.size(src_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(src_img_, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(src_img_, temp)
        src_skel = cv2.bitwise_or(src_skel, temp)
        src_img_ = eroded.copy()

        zeros = size - cv2.countNonZero(src_img_)
        if zeros == size:
            done = True

    tag_skel = np.zeros(tag_img.shape, np.uint8)
    size = np.size(tag_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(tag_img_, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(tag_img_, temp)
        tag_skel = cv2.bitwise_or(tag_skel, temp)
        tag_img_ = eroded.copy()

        zeros = size - cv2.countNonZero(tag_img_)
        if zeros == size:
            done = True

    cv2.imshow("src edge", src_edges)
    cv2.imshow("tag edge", tag_edges)

    cv2.imshow("src skel", src_skel)
    cv2.imshow("tag skel", tag_skel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()