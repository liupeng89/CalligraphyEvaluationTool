import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    src_path = "src_resize.png"
    tag_path = "tag_resize.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

    # src image polynomical prediction of degree 1
    src_x_list = []
    src_y_list = []
    for y in range(src_img.shape[0]):
        for x in range(src_img.shape[1]):
            if src_img[y][x] == 0.0:
                src_x_list.append(x)
                src_y_list.append(y)

    # z[0] = k, z[1] = b
    z = np.polyfit(src_x_list, src_y_list, 1)
    print(z)

    # horizon or vertical
    k = z[0]
    b = z[1]
    if k >= -1 and k <= 1:
        # horizon
        x0 = 0; xn = src_img.shape[1] - 1
        y0 = int(k * x0 + b)
        yn = int(k * xn + b)
        src_img_rgb = cv2.line(src_img_rgb, (y0, x0), (yn, xn), (0, 0, 255), 2)
    elif k < -1 or k > 1:
        # vertical
        y0 = 0; yn = src_img.shape[0] - 1
        x0 = int(k * y0 + b)
        xn = int(k * yn + b)
        src_img_rgb = cv2.line(src_img_rgb, (y0, x0), (yn, xn), (0, 0, 255), 2)

    # target image polynomical prediction of degree 1
    tag_x_list = []
    tag_y_list = []
    for y in range(tag_img.shape[0]):
        for x in range(tag_img.shape[1]):
            if tag_img[y][x] == 0.0:
                    tag_x_list.append(x)
                    tag_y_list.append(y)

    # z[0] = k, z[1] = b
    z = np.polyfit(tag_x_list, tag_y_list, 1)
    print(z)

    # horizon or vertical
    k = z[0]
    b = z[1]
    if k >= -1 and k <= 1:
        # horizon
        x0 = 0;
        xn = tag_img.shape[1] - 1
        y0 = int(k * x0 + b)
        yn = int(k * xn + b)
        tag_img_rgb = cv2.line(tag_img_rgb, (y0, x0), (yn, xn), (0, 0, 255), 2)
    elif k < -1 or k > 1:
        # vertical
        y0 = 0;
        yn = tag_img.shape[0] - 1
        x0 = int(k * y0 + b)
        xn = int(k * yn + b)
        tag_img_rgb = cv2.line(tag_img_rgb, (y0, x0), (yn, xn), (0, 0, 255), 2)

    cv2.imshow("src", src_img_rgb)
    cv2.imshow("tag", tag_img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








if __name__ == '__main__':
    main()