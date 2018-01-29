import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    src_path = "../chars/src_dan_svg_simple_resized.png"
    tag_path = "../chars/tag_dan_svg_simple_resized.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # x-axis and y-axis statistics histogram
    src_x_hist = np.zeros(src_img.shape[1])
    src_y_hist = np.zeros(src_img.shape[0])

    tag_x_hist = np.zeros(tag_img.shape[1])
    tag_y_hist = np.zeros(tag_img.shape[0])

    for y in range(src_img.shape[0]):
        for x in range(src_img.shape[1]):
            if src_img[y][x] == 0:
                src_y_hist[y] += 1
                src_x_hist[x] += 1

    for y in range(tag_img.shape[0]):
        for x in range(tag_img.shape[1]):
            if tag_img[y][x] == 0:
                tag_y_hist[y] += 1
                tag_x_hist[x] += 1

    # print(src_x_hist)
    # print(src_y_hist)

    plt.subplot(221);
    plt.plot(src_x_hist)
    plt.title("X axis")
    plt.ylabel("Source image")
    plt.subplot(222);
    plt.plot(src_y_hist)
    plt.title("Y axis")
    plt.subplot(223);
    plt.ylabel("Target image")
    plt.plot(tag_x_hist)
    plt.subplot(224);
    plt.plot(tag_y_hist)


    # plt.subplot(121);
    # line1, = plt.plot(src_x_hist, label="source")
    # line2, = plt.plot(tag_x_hist, label="target")
    # plt.xlabel("X axis")
    # plt.legend(handles=[line1, line2], loc=2)
    #
    # plt.subplot(122);
    # line3, = plt.plot(src_y_hist, label="source")
    # line4, = plt.plot(tag_y_hist, label="target")
    # plt.xlabel("Y axis")
    #
    # plt.legend(handles=[line3, line4], loc=2)

    plt.show()



if __name__ == '__main__':
    main()