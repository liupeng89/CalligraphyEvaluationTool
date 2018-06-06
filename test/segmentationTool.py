import cv2
import numpy as np
from utils.Functions import getAllMiniBoundingBoxesOfImage
import copy

WIDTH_THRESHOLD = 15
HEIGHT_THRESHOLD = 15
SCALE_RATIO = 0.4


def main():
    src_path = '../calligraphys/hurulin1.jpg'

    img = cv2.imread(src_path, 0)

    img_rgb = cv2.imread(src_path)

    img = cv2.resize(img, (0, 0), fx=SCALE_RATIO, fy=SCALE_RATIO)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    img_rgb = cv2.resize(img_rgb, (0, 0), fx=SCALE_RATIO, fy=SCALE_RATIO)

    img_rgb_noinside = copy.deepcopy(img_rgb)
    img_rgb_nointersected = copy.deepcopy(img_rgb)

    # image processing

    # image binary
    _, img_bit = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print(img_bit.shape)

    # inverting color
    # img_bit = 255 - img_bit

    # get all bounding boxes
    boxes = getAllMiniBoundingBoxesOfImage(img_bit)
    print("boxes len:%d" % len(boxes))

    # filter the boxes, and remove the biggest and too small boxes
    boxes_filtered = []
    for box in boxes:
        if box[2] < WIDTH_THRESHOLD * SCALE_RATIO or box[3] < HEIGHT_THRESHOLD * SCALE_RATIO:
            continue

        if box[2] >= img_bit.shape[0] or box[3] >= img_bit.shape[1]:
            continue

        boxes_filtered.append(box)
    print("after filtered boxes len: %d" % len(boxes_filtered))

    # removed contained rectangles
    inside_id = []
    for i in range(len(boxes_filtered)):
        ri_x = boxes_filtered[i][0]
        ri_y = boxes_filtered[i][1]
        ri_w = boxes_filtered[i][2]
        ri_h = boxes_filtered[i][3]

        for j in range(len(boxes_filtered)):
            if i == j or j in inside_id:
                continue
            rj_x = boxes_filtered[j][0]
            rj_y = boxes_filtered[j][1]
            rj_w = boxes_filtered[j][2]
            rj_h = boxes_filtered[j][3]

            # rect_j  inside rect_i
            if ri_x <= rj_x and ri_y <= rj_y and ri_x + ri_w >= rj_x + rj_w and ri_y + ri_h >= rj_y + rj_h:
                if j not in inside_id:
                    inside_id.append(j)
            elif rj_x <= ri_x and rj_y <= ri_y and rj_x + rj_w >= ri_x + ri_w and rj_y + rj_h >= ri_y + ri_h:
                if i not in inside_id:
                    inside_id.append(i)
    print("insie id len: %d " % len(inside_id))

    boxes_noinside = []
    for i in range(len(boxes_filtered)):
        if i in inside_id:
            continue
        boxes_noinside.append(boxes_filtered[i])
    print("no inside box len: %d" % len(boxes_noinside))

    # add rectangles to original images

    for box in boxes_noinside:
        img_rgb_noinside = cv2.rectangle(img_rgb_noinside, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 1)

    # merge the intersected rectangles
    intersected_id_list = []
    for i in range(len(boxes_noinside)):
        intersected_item = []
        rect_i = boxes_noinside[i]
        intersected_item.append(i)
        for j in range(len(boxes_noinside)):
            if i == j:
                continue
            rect_j = boxes_noinside[j]
            if isIntersectedOfTwoRectangles(rect_i, rect_j):
                intersected_item.append(j)
        intersected_id_list.append(intersected_item)
    print("Before merge the id list")
    print(intersected_id_list)

    # merge the id list
    intersected_id_list_merged = []
    used_id = []
    for i in range(len(intersected_id_list)):
        if i in used_id:
            continue
        # no intersected with others
        # if len(intersected_id_list[i]) == 1:
        #     intersected_id_list_merged.append(intersected_id_list[i])
        #     continue

        new_id = intersected_id_list[i]
        # intersected with others
        for j in range(i+1, len(intersected_id_list)):
            if len(set(intersected_id_list[i]).intersection(set(intersected_id_list[j]))) == 0:
                continue
            # intersected
            new_id = list(set(new_id).union(set(intersected_id_list[j])))
            used_id.append(j)
        intersected_id_list_merged.append(new_id)

    print("After merge")
    print(intersected_id_list_merged)

    boxes_nointersected = []

    for item in intersected_id_list_merged:
        rt_x, rt_y, rt_w, rt_h = combineRectangles(boxes_noinside, item)
        boxes_nointersected.append([rt_x, rt_y, rt_w, rt_h])

    print(len(boxes_nointersected))

    for box in boxes_nointersected:
        img_rgb_nointersected = cv2.rectangle(img_rgb_nointersected, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)

    cv2.imshow("noinside", img_rgb_noinside)
    cv2.imshow("nointersected", img_rgb_nointersected)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def isIntersectedOfTwoRectangles(rect1, rect2):
    """
    Detect whether two rectangles are intersected with each other or not.
    :param rect1: rectangle 1
    :param rect2: rectangle 2
    :return: intersected or not.
    """
    if rect1 is None or rect2 is None:
        return False
    if rect1[0] < rect2[0]+rect2[2] and rect1[0]+rect1[2] > rect2[0] and rect1[1] < rect2[1]+rect2[3] and rect1[1]+rect1[3] > rect2[1]:
        return True
    return False


def combineRectangles(rectangles, rect_list):
    """
    Combining rectangles together.
    :param rectangles: list of rectangles.
    :param rect_list: list of index of rectangles in rectangles list.
    :return: new combined rectangle.
    """
    if rectangles is None or len(rect_list) == 0:
        return None

    new_rect_x1 = rectangles[rect_list[0]][0]
    new_rect_y1 = rectangles[rect_list[0]][1]
    new_rect_x2 = rectangles[rect_list[0]][2] + rectangles[rect_list[0]][0]
    new_rect_y2 = rectangles[rect_list[0]][3] + rectangles[rect_list[0]][1]

    for id in range(len(rect_list)):
        rect_x1 = rectangles[rect_list[id]][0]
        rect_y1 = rectangles[rect_list[id]][1]
        rect_x2 = rectangles[rect_list[id]][2] + rectangles[rect_list[id]][0]
        rect_y2 = rectangles[rect_list[id]][3] + rectangles[rect_list[id]][1]

        x1 = min(new_rect_x1, rect_x1)
        y1 = min(new_rect_y1, rect_y1)

        x2 = max(new_rect_x2, rect_x2)
        y2 = max(new_rect_y2, rect_y2)

        new_rect_x1 = x1
        new_rect_y1 = y1
        new_rect_x2 = x2
        new_rect_y2 = y2

    return new_rect_x1, new_rect_y1, new_rect_x2 - new_rect_x1, new_rect_y2 - new_rect_y1


if __name__ == '__main__':
    main()