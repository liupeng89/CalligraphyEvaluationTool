import cv2
from utils.Functions import resizeImages


def main():

    src_path = "../characters/src_dan_processed.png"
    tag_path = "../characters/tag_dan_processed.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    src_img, tag_img = resizeImages(src_img, tag_img)

    print(src_img.shape)
    print(tag_img.shape)

    cv2.imshow("source", src_img)
    cv2.imshow("target", tag_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()