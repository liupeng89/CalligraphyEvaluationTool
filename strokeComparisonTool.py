import cv2
import numpy as np
from utils.Functions import calculateBoundingBox, getContourOfImage, getSkeletonOfImage, getRotatedMinimumBoundingBox


def main():
    src_path = "../strokes/src_strokes7.png"
    tag_path = "../strokes/tag_strokes7.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    # threshold
    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # get the minimum bounding boxs
    src_box = calculateBoundingBox(src_img)
    tag_box = calculateBoundingBox(tag_img)

    # get the region of strokes
    src_region = src_img[src_box[1]-5:src_box[1]+src_box[3]+5, src_box[0]-5: src_box[0]+src_box[2]+5]
    tag_region = tag_img[tag_box[1]-5:tag_box[1]+tag_box[3]+5, tag_box[0]-5: tag_box[0]+tag_box[2]+5]


    # get the contour of storkes based on the Canny algorithm

    src_edge = getContourOfImage(src_region)
    tag_edge = getContourOfImage(tag_region)

    cv2.imshow("src edge", src_edge)
    cv2.imshow("tag edge", tag_edge)


    # get the skeletons of strokes based on the thinning algorithm

    src_skeleton = getSkeletonOfImage(src_region)
    tag_skeleton = getSkeletonOfImage(tag_region)

    cv2.imshow("src skel", src_skeleton)
    cv2.imshow("tag skel", tag_skeleton)

    # split the strokes based on the rule: begin, middle and the end parts.
    src_regions = splitStrokes(src_region, type="LongHeng")
    tag_regions = splitStrokes(tag_region, type="LongHeng")

    print('len src regions: %d' % len(src_regions))
    print('len tag regions: %d' % len(tag_regions))

    cv2.imshow('src begin', src_regions[0])
    cv2.imshow('src middle', src_regions[1])
    cv2.imshow('src end', src_regions[2])

    cv2.imshow("src", src_region )
    cv2.imshow("tag", tag_region)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    Split the strokes based on the type of strokes.
    Type 1: Long_Heng, crop to three parts: begin(1/5), middle (3/5) and end (1/5)
'''
def splitStrokes(image, type):
    if image is None:
        return None
    result = []
    # Type 1: LongHeng, the 1/5 part is the begining, the last 1/5 part is the ending, and the rest part is the middle.
    if type == 'LongHeng':
        range = max(image.shape[0], image.shape[1])

        # Begining part
        begin_region = image[:, 0:int(range * 1/5)]

        # Middle part
        middle_region = image[:, int(range * 1/5): int(range * 4/5)]

        # Ending part
        end_region = image[:, int(range * 4/5): range]

        result.append(begin_region)
        result.append(middle_region)
        result.append(end_region)

        return result

        # begin= 1/5 * length

    # Type 2: MiddleHeng, the 1/4 pare is the begining, the last 1/4 part is the ending, and the rest part is the middle.
    if type == 'MiddleHeng':
        pass


    # Type 3: ShortHeng, the 1/2 part is the begining, the rest 1/2 part is the ending, and no middle part
    if type == 'ShortHeng':
        pass










if __name__ == '__main__':
    main()
