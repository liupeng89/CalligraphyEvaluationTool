# coding: utf-8


def filterBoxWithWidth(boxes, threshold_width):
    """
    Filter boxes with threshold of width. Too small box should be removed!
    :param boxes:
    :param threshold_width:
    :return:
    """
    if boxes is None:
        return
    boxes_ = []
    for box in boxes:
        if box[2] < threshold_width and box[3] < threshold_width:
            continue
        boxes_.append(box)

    return boxes_


def removeContainedBoxes(boxes):
    """
    Remove the box inside other box.
    :param boxes:
    :return:
    """
    if boxes is None:
        return

    inside_id = []
    for i in range(len(boxes)):
        ri_x = boxes[i][0]; ri_y = boxes[i][1]; ri_w = boxes[i][2]; ri_h = boxes[i][3]

        for j in range(len(boxes)):
            if i == j or j in inside_id:
                continue
            rj_x = boxes[j][0]; rj_y = boxes[j][1]; rj_w = boxes[j][2]; rj_h = boxes[j][3]

            # rect_j  inside rect_i
            if ri_x <= rj_x and ri_y <= rj_y and ri_x + ri_w >= rj_x + rj_w and ri_y + ri_h >= rj_y + rj_h:
                if j not in inside_id:
                    inside_id.append(j)
            elif rj_x <= ri_x and rj_y <= ri_y and rj_x + rj_w >= ri_x + ri_w and rj_y + rj_h >= ri_y + ri_h:
                if i not in inside_id:
                    inside_id.append(i)

    print("inside id len: %d" % len(inside_id))

    boxes_noinside = []
    for i in range(len(boxes)):
        if i in inside_id:
            continue
        boxes_noinside.append(boxes[i])
    print("no inside boxes len: %d" % len(boxes_noinside))
    del inside_id
    return boxes_noinside

