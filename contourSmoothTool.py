import cv2
import numpy as np
from utils.Functions import getContourOfImage, order_points

def main():

    # load image
    img_path = "../templates/stroke_dan.png"

    img = cv2.imread(img_path, 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    # get contour of image
    contour = getContourOfImage(img)
    contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)

    # order the contour points
    contour_points_clockwise = order_points(contour)
    # contour_points_counter_clockwise = order_points(contour, isClockwise=False)
    print("clock :%d" % len(contour_points_clockwise))
    # print("counter clock: %d" % len(contour_points_counter_clockwise))

    contour_rgb_clock = contour_rgb.copy()
    # contour_rgb_counter_clock = contour_rgb.copy()

    for id, pt in enumerate(contour_points_clockwise):
        if id % 100 == 0:
            cv2.circle(contour_rgb_clock, pt, 2, (0,255,0),-1)
            cv2.putText(contour_rgb_clock, str(id), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)

    # for id, pt in enumerate(contour_points_counter_clockwise):
    #     if id % 100 == 0:
    #         cv2.circle(contour_rgb_counter_clock, pt, 2, (0,255,0),-1)
    #         cv2.putText(contour_rgb_counter_clock, str(id), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)


    # get key points on contour
    corners = cv2.goodFeaturesToTrack(contour, 10, 0.01, 10)
    corners = np.int0(corners)

    index = 0
    corner_points_ = []
    for i in corners:
        x,y = i.ravel()
        corner_points_.append((x, y))
        cv2.circle(contour_rgb, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(contour_rgb, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)
        index += 1

    corner_points = []
    for pt in contour_points_clockwise:
        if pt in corner_points_:
            corner_points.append(pt)
    print("corner points len: %d" % len(corner_points))

    cv2.imshow("src", img)
    cv2.imshow("contour", contour)
    cv2.imshow("corners", contour_rgb)
    cv2.imshow("contour clock", contour_rgb_clock)
    # cv2.imshow("contour counter clock", contour_rgb_counter_clock)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()