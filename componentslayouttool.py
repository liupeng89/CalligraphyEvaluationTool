import numpy as np
import cv2


def main():
    src_path = "../chars/src_dan_svg_simple_resized.png"
    tag_path = "../chars/tag_dan_svg_simple_resized.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    src_img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
    tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

    # src component rects
    _, src_contours, _ = cv2.findContours(src_img, 1, 2)
    # for src_contour in src_contours:
    #     x, y, w, h = cv2.boundingRect(src_contour)
    #     if w > 0.95 * src_img.shape[0] and h > 0.95 * src_img.shape[1]:
    #         continue
    #
    #     src_img_rgb = cv2.rectangle(src_img_rgb, (x,y), (x+w, y+h), (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(src_contours[0])
    src_img_rgb = cv2.rectangle(src_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    c0_x = x + int(w / 2)
    c0_y = y + int(h / 2)
    # x, y, w, h = cv2.boundingRect(src_contours[1])
    # src_img_rgb = cv2.rectangle(src_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(src_contours[2])
    src_img_rgb = cv2.rectangle(src_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    c2_x = x + int(w/2)
    c2_y = y + int(h/2)

    x, y, w, h = cv2.boundingRect(src_contours[3])
    src_img_rgb = cv2.rectangle(src_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    c3_x = x + int(w/2)
    c3_y = y + int(h/2)

    src_img_rgb = cv2.line(src_img_rgb, (c0_x, c0_y), (c2_x, c2_y), (0, 0, 255), 2)
    src_img_rgb = cv2.line(src_img_rgb, (c0_x, c0_y), (c3_x, c3_y), (0, 0, 255), 2)
    src_img_rgb = cv2.line(src_img_rgb, (c3_x, c3_y), (c2_x, c2_y), (0, 0, 255), 2)



    # tag component rects
    _, tag_contours, _ = cv2.findContours(tag_img, 1, 2)
    # for tag_contour in tag_contours:
    #     x, y, w, h = cv2.boundingRect(tag_contour)
    #     if w > 0.95 * tag_img.shape[0] and h > 0.95 * tag_img.shape[1]:
    #         continue
    #
    #     tag_img_rgb = cv2.rectangle(tag_img_rgb, (x,y), (x+w, y+h), (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(tag_contours[0])
    tag_img_rgb = cv2.rectangle(tag_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    c0_x = x+ int(w/2)
    c0_y = y + int(h/2)
    x1, y1, w1, h1 = cv2.boundingRect(tag_contours[1])
    # tag_img_rgb = cv2.rectangle(tag_img_rgb, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    # x, y, w, h = cv2.boundingRect(tag_contours[2])
    # tag_img_rgb = cv2.rectangle(tag_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x3, y3, w3, h3 = cv2.boundingRect(tag_contours[3])
    tag_img_rgb = cv2.rectangle(tag_img_rgb, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 0), 2)
    c3_x = x3 + int(w3/2)
    c3_y = y3 + int(h3/2)
    x4, y4, w4, h4 = cv2.boundingRect(tag_contours[4])
    # tag_img_rgb = cv2.rectangle(tag_img_rgb, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)

    new_x = min(x1, x4)
    new_y = min(y1, y4)
    new_w = max(x1+w1, x4+w4) - new_x
    new_h = max(y1+h1, y4+h4) - new_y
    tag_img_rgb = cv2.rectangle(tag_img_rgb, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
    cnew_x = new_x + int(new_w/2)
    cnew_y = new_y + int(new_h/2)

    tag_img_rgb = cv2.line(tag_img_rgb, (c0_x, c0_y), (c3_x, c3_y), (0, 0, 255), 2)
    tag_img_rgb = cv2.line(tag_img_rgb, (c3_x, c3_y), (cnew_x, cnew_y), (0, 0, 255), 2)
    tag_img_rgb = cv2.line(tag_img_rgb, (c0_x, c0_y), (cnew_x, cnew_y), (0, 0, 255), 2)


    cv2.imshow("src", src_img_rgb)
    cv2.imshow("tag", tag_img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()