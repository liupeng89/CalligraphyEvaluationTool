# coding: utf-8

import os
import cv2
import numpy as np

from utils.Functions import splitConnectedComponents


def main():
    charset_dir = "../simkaidataset6958"

    char_list = []
    char_names = []
    for fl in os.listdir(charset_dir):
        if ".jpg" in fl:
            char_list.append(fl)
            char_names.append(os.path.splitext(fl)[0])

    print(char_list)
    print(char_names)

    radicals_path = charset_dir + "/radicals"
    strokes_path = charset_dir + "/strokes"

    for i in range(len(char_list)):
        print("index: %d" % i)
        f_path = os.path.join(charset_dir, char_list[i])
        f_name = char_names[i]

        img_gray = cv2.imread(f_path, 0)
        _, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        boxes = splitConnectedComponents(img_gray)
        print(len(boxes))

        radical_path = str(os.path.join(radicals_path, f_name))
        if not os.path.exists(radical_path):
            os.mkdir(radical_path)
        for bi in range(len(boxes)):
            r_path = radical_path + "/radicals_" + str(bi) + ".jpg"
            cv2.imwrite(r_path, boxes[bi])


if __name__ == '__main__':
    main()