import cv2
import numpy as np
from utils.Functions import calculateBoundingBox


def main():
    src_path = "../strokes/src_strokes1.png"
    tag_path = "../strokes/tag_strokes1.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    # get the minimum bounding boxs
    src_box = calculateBoundingBox(src_img)
    tag_box = calculateBoundingBox(tag_img)

    # get the region of strokes
    src_region = src_img[src_box[1]-5:src_box[1]+src_box[3]+5, src_box[0]-5: src_box[0]+src_box[2]+5]
    tag_region = tag_img[tag_box[1]-5:tag_box[1]+tag_box[3]+5, tag_box[0]-5: tag_box[0]+tag_box[2]+5]




    cv2.imshow("src", src_region )
    cv2.imshow("tag", tag_region)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
