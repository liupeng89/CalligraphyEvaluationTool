import os
import cv2
import numpy as np
from utils.Functions import getSingleMaxBoundingBoxOfImage


def main():
    base_path = "../templates/templates_comparison"

    temp_roots = [f for f in os.listdir(base_path) if "." not in f]

    for temp_rt in temp_roots:
        temp_path = base_path + "/" + temp_rt + "/char/" + temp_rt + ".png"
        print(temp_path)

        if not os.path.exists(temp_path):
            continue

        temp_img = cv2.imread(temp_path, 0)
        _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)

        x, y, w, h = getSingleMaxBoundingBoxOfImage(temp_img)

        crop_img = temp_img[y: y+h, x: x+w]

        crop_path = base_path + "/" + temp_rt + "/char/" + temp_rt + "_crop.png"
        cv2.imwrite(crop_path, crop_img)

        c0_x = int(crop_img.shape[1] / 2)
        c0_y = int(crop_img.shape[0] / 2)

        new_w = max(w, h) + int(0.1 * max(w, h))
        new_h = new_w

        c1_x = int(new_w / 2)
        c1_y = int(new_h / 2)

        # offset
        offset_x = c1_x - c0_x
        offset_y = c1_y - c0_y

        new_img = np.ones((new_w, new_h)) * 255
        new_img = np.array(new_img, dtype=np.uint8)

        for y in range(crop_img.shape[0]):
            for x in range(crop_img.shape[1]):
                new_img[y+offset_y][x+offset_x] = crop_img[y][x]

        resize_path = base_path + "/" + temp_rt + "/char/" + temp_rt + "_resize.png"

        cv2.imwrite(resize_path, new_img)


if __name__ == '__main__':
    main()

    # strokes_path = "../templates/templates/ben/strokes"
    #
    # files = [f for f in os.listdir(strokes_path) if '.png' in f]
    #
    # print(files)
    #
    # for file in files:
    #     file_path = strokes_path + "/" + file
    #
    #     img = cv2.imread(file_path, 0)
    #
    #     _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #
    #     h, w = img.shape[0], img.shape[1]
    #
    #     c0_x = int(w / 2)
    #     c0_y = int(h / 2)
    #
    #     new_w = max(w, h) + int(0.1 * max(w, h))
    #     new_h = new_w
    #
    #     c1_x = int(new_w / 2)
    #     c1_y = int(new_h / 2)
    #
    #     # offset
    #     offset_x = c1_x - c0_x
    #     offset_y = c1_y - c0_y
    #
    #     new_img = np.ones((new_w, new_h)) * 255
    #     new_img = np.array(new_img, dtype=np.uint8)
    #
    #     for y in range(h):
    #         for x in range(w):
    #             new_img[y+offset_y][x+offset_x] = img[y][x]
    #
    #     save_path = strokes_path + "/_" + file
    #     cv2.imwrite(save_path, new_img)






