import numpy as np
import cv2
from utils.Functions import  getCenterOfGravity


def main():
    src_path = "src_resize.png"
    tag_path = "tag_resize.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

    # center of gravity of source
    src_cog_x, src_cog_y = getCenterOfGravity(src_img)
    src_img_rgb = cv2.circle(src_img_rgb, (src_cog_y, src_cog_x), 4, (0, 0, 255), -1)
    print("src : (%d, %d)" % (src_cog_x, src_cog_y))

    # center of gravity of target
    tag_cog_x, tag_cog_y = getCenterOfGravity(tag_img)
    tag_img_rgb = cv2.circle(tag_img_rgb, (tag_cog_y, tag_cog_x), 4, (0, 0, 255), -1)
    print("tag : (%d, %d)" % (tag_cog_x, tag_cog_y))

    cv2.imshow("src", src_img_rgb)
    cv2.imshow("tag", tag_img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()