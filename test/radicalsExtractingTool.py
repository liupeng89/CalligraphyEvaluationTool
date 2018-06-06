import cv2
import numpy as np

from utils.Functions import splitConnectedComponents

def main():
    img_path = "ben.png"

    img = cv2.imread(img_path, 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # extract radicals of character
    radicals = splitConnectedComponents(img)

    print("radicals len: %d" % len(radicals))

    for id, rd in enumerate(radicals):
        cv2.imshow("id:"+str(id), rd)


    cv2.imshow("src", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()