import cv2
from utils.Functions import resizeImages, coverTwoImages, shiftImageWithMaxCR, calculateCR, addIntersectedFig, addSquaredFig


def main():
    # src_path = "../chars/src_dan_svg_simple_resized.png"
    # tag_path = "../chars/tag_dan_svg_simple_resized.png"
    src_path = "../strokes/src_strokes4.png"
    tag_path = "../strokes/tag_strokes4.png"

    # src_path = "../strokes/src_strokes1.png"
    # tag_path = "../strokes/tag_strokes1.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    ret, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # resize
    src_img, tag_img = resizeImages(src_img, tag_img)

    # Threshold
    ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    ret, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # cv2.imwrite("src_resize.png", src_img)
    # cv2.imwrite("tag_resize.png", tag_img)

    # Cover Images

    coverage_img = coverTwoImages(src_img, tag_img)
    cr = calculateCR(src_img, tag_img)
    print("No shifting cr: %f" % cr)

    # coverage_img = addIntersectedFig(coverage_img)
    # coverage_img = addSquaredFig(coverage_img)

    # Shift images with max CR
    new_tag_img = shiftImageWithMaxCR(src_img, tag_img)

    # Cover images
    coverage_img1 = coverTwoImages(src_img, new_tag_img)
    cr = calculateCR(src_img, new_tag_img)
    print("Shifting cr: %f" % cr)

    coverage_img_ = addIntersectedFig(coverage_img)
    coverage_img1_ = addIntersectedFig(coverage_img1)

    cv2.imshow("coverage img", coverage_img_)
    cv2.imshow("new coverage img", coverage_img1_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()