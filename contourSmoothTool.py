import cv2
import numpy as np


def main():
    img_path = "../templates/stroke_2.png"

    img = cv2.imread(img_path, 0)

    cv2.imshow("src", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()