import numpy as np
import cv2
from utils.Functions import getConnectedComponents


def main():
    src_path = "../chars/src_dan_svg_simple_resized.png"
    tag_path = "../chars/tag_fu_processed.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_img = np.uint8(src_img)
    tag_img = np.uint8(tag_img)

    # grayscale to rgb
    src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

    # connectivity
    components = getConnectedComponents(tag_img)
    print("components len: %d" % len(components))

    # Check C-points exist or not. if no C-points, the single stroke. But C-points exist, several strokes

    for idx, components in enumerate(components):
        win_name = "src " + str(idx)
        cv2.imshow(win_name, components)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




    # for lab in range(num_labels):
    #
    #     x0 = stats[lab, cv2.CC_STAT_LEFT]
    #     y0 = stats[lab, cv2.CC_STAT_TOP]
    #     w = stats[lab, cv2.CC_STAT_WIDTH]
    #     h = stats[lab, cv2.CC_STAT_HEIGHT]
    #
    #     if x0 == 0 and y0 == 0:
    #         continue
    #     src_img_rgb = cv2.rectangle(src_img_rgb, (x0, y0), (x0+w, y0+h), (0,255,0), 2)
    #
    # cv2.imshow("source", fig1)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    main()