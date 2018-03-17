import cv2
import numpy


from utils.Functions import calculateBoundingBox, calculateCR


def main():
    pass


if __name__ == '__main__':
    # main()
    src_path = "../templates/ben.png"
    temp_base_path = "../templates/stroke_"

    src_img = cv2.imread(src_path, 0)
    # temp_img = cv2.imread(temp_path, 0)
    # w, h = temp_img.shape

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    # _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("src", src_img)
    # cv2.imshow("temp", temp_img)

    # bounding box
    src_x, src_y, src_w, src_h = calculateBoundingBox(src_img)
    # temp_x, temp_y, temp_w, temp_h = calculateBoundingBox(temp_img)

    src_window = src_img[src_y: src_y+src_h, src_x:src_x+src_w]
    # temp_window = temp_img[temp_y:temp_y+temp_h, temp_x:temp_x+temp_w]

    # temp_path = temp_base_path + "2_2.png"
    # temp_img = cv2.imread(temp_path, 0)
    # _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)
    #
    # temp_x, temp_y, temp_w, temp_h = calculateBoundingBox(temp_img)
    # temp_window = temp_img[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w]
    #
    # src_part = src_window[185:185+temp_h, 306:306+temp_w]
    #
    # src_part1 = src_window[455:455+temp_h, 165:165+temp_w]
    #
    # cv2.imshow("src part", src_part)
    # cv2.imshow("temp", temp_window)
    # cv2.imshow("src 2", src_part1)
    #
    # cr = calculateCR(src_part, temp_window)
    #
    # cr1 = calculateCR(src_part1, temp_window)
    #
    # print(cr)
    # print(cr1)




    src_window_rgb = cv2.cvtColor(src_window, cv2.COLOR_GRAY2RGB)

    for id in range(2, 3):
        temp_path = temp_base_path + str(id) + ".png"
        print(temp_path)
        temp_img = cv2.imread(temp_path, 0)
        _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)

        temp_img = cv2.resize(temp_img, src_img.shape)

        temp_x, temp_y, temp_w, temp_h = calculateBoundingBox(temp_img)
        temp_window = temp_img[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w]

        max_cr = -1000000000.0
        loc_x = 0; loc_y = 0
        for y in range(src_h-temp_h):
            for x in range(src_w-temp_w):
                win_ = src_window[y:y+temp_h, x:x+temp_w]
                cr = calculateCR(win_, temp_window)
                print(cr)
                if cr > max_cr:
                    print("cr: %f" % cr)
                    max_cr = cr
                    loc_x = x
                    loc_y = y
        print("loc: (%d, %d)" % (loc_x, loc_y))
        print("max cr: %f" % max_cr)

        src_window_rgb = cv2.rectangle(src_window_rgb, (loc_x, loc_y), (loc_x+temp_w, loc_y+temp_h), (0,255,0), 1)

    cv2.imshow("src window", src_window)
    cv2.imshow("temp window", temp_window)
    cv2.imshow("src window rgb", src_window_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





