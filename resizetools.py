import cv2
from utils.Functions import resizeImages


def main():

    src_path = "../chars/src_dan_svg_simple.png"
    tag_path = "../chars/tag_dan_svg_simple.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    src_img, tag_img = resizeImages(src_img, tag_img)

    print(src_img.shape)
    print(tag_img.shape)

    cv2.imwrite('../chars/src_dan_svg_simple_resized.png', src_img)
    cv2.imwrite('../chars/tag_dan_svg_simple_resized.png', tag_img)

    cv2.imshow("source", src_img)
    cv2.imshow("target", tag_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()