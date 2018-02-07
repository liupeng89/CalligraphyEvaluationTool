import cv2
import numpy as np
from utils.Functions import getBoundingBoxes
import math

def main():
    scale_ratio = 0.4
    # src_path = '../calligraphys/yaomengqi1.jpg'
    # src_path = '../calligraphys/ouyangxun1.jpg'
    src_path = '../calligraphys/hurulin1.jpg'

    img = cv2.imread(src_path, 0)

    img_rgb = cv2.imread(src_path)

    img = cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    img_rgb = cv2.resize(img_rgb, (0,0), fx=scale_ratio, fy=scale_ratio)

    # image processing

    # image binary
    _, img_bit = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print(img_bit.shape)

    # inverting color
    # img_bit = 255 - img_bit

    # get all bounding boxes
    boxes = getBoundingBoxes(img_bit)
    print(len(boxes))

    boxes_ = []

    for box in boxes:
        if box[2] < 30*scale_ratio or box[3] < 30*scale_ratio:
            continue

        if box[2] >= img_bit.shape[0] or box[3] >= img_bit.shape[1]:
            continue

        boxes_.append(box)

    print(len(boxes_))

    # remove inside rectangles
    inside_id = []
    for i in range(len(boxes_)):
        ri_x = boxes_[i][0]
        ri_y = boxes_[i][1]
        ri_w = boxes_[i][2]
        ri_h = boxes_[i][3]

        for j in range(len(boxes_)):
            if i == j or j in inside_id:
                continue
            rj_x = boxes_[j][0]
            rj_y = boxes_[j][1]
            rj_w = boxes_[j][2]
            rj_h = boxes_[j][3]

            # rect_j  inside rect_i
            if ri_x <= rj_x and ri_y <= rj_y and ri_x+ri_w >= rj_x+rj_w and ri_y+ri_h >= rj_y+rj_h:
                if j not in inside_id:
                    inside_id.append(j)
            elif rj_x <= ri_x and rj_y <= ri_y and rj_x+rj_w >= ri_x+ri_w and rj_y+rj_h >= ri_y+ri_h:
                if i not in inside_id:
                    inside_id.append(i)

    print("inside id len: %d" % len(inside_id))

    boxes_noinsde = []
    for i in range(len(boxes_)):
        if i in inside_id:
            continue
        boxes_noinsde.append(boxes_[i])
    print("no inside box len: %d" % len(boxes_noinsde))

    # add rectangles to original images
    for box in boxes_noinsde:
        img_rgb = cv2.rectangle(img_rgb, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 1)

    # handle the intersect rectangles

    # add line of center of rectangles
    dist_threshod = 70 * scale_ratio
    for i in range(len(boxes_noinsde)):
        ct_rect_i = getCenterOfRectangles(boxes_noinsde[i])

        for j in range(len(boxes_noinsde)):
            if i == j:
                continue
            ct_rect_j = getCenterOfRectangles(boxes_noinsde[j])

            dist = math.sqrt((ct_rect_j[0] - ct_rect_i[0]) * (ct_rect_j[0] - ct_rect_i[0]) + (ct_rect_j[1] - ct_rect_i[1]) * (ct_rect_j[1] - ct_rect_i[1]))

            if dist > dist_threshod:
                continue
            img_rgb = cv2.line(img_rgb, (ct_rect_i[0], ct_rect_i[1]), (ct_rect_j[0], ct_rect_j[1]), (0, 255, 0), 1)

    # cluster rectangles based on the distance
    rect_clustor = []
    for i in range(len(boxes_noinsde)):
        rect_item = []
        rect_item.append(i)

        ct_rect_i = getCenterOfRectangles(boxes_noinsde[i])

        for j in range(len(boxes_noinsde)):
            ct_rect_j = getCenterOfRectangles(boxes_noinsde[j])

            dist = math.sqrt( (ct_rect_j[0] - ct_rect_i[0]) * (ct_rect_j[0] - ct_rect_i[0]) + (ct_rect_j[1] - ct_rect_i[1]) * (
                            ct_rect_j[1] - ct_rect_i[1]))
            if dist <= dist_threshod and j not in rect_item:
                rect_item.append(j)
        rect_clustor.append(rect_item)

    print(rect_clustor)

    # merge the clustor
    final_clustor = []
    used_index = []
    for i in range(len(rect_clustor)):
        if i in used_index:
            continue
        new_clustor = rect_clustor[i]
        # merge
        for j in range(i+1, len(rect_clustor)):
            if len(set(new_clustor).intersection(set(rect_clustor[j]))) == 0:
                continue
            new_clustor = list(set(new_clustor).union(set(rect_clustor[j])))
            used_index.append(j)
        final_clustor.append(new_clustor)

    print(final_clustor)

    for i in range(len(final_clustor)):

        new_rect = combineRectangles(boxes_noinsde, final_clustor[i])
        img_rgb = cv2.rectangle(img_rgb, (new_rect[0], new_rect[1]), (new_rect[0] + new_rect[2], new_rect[1] + new_rect[3]), (0, 0, 255), 1)






    # the projecting histogram on X-axis (columns)
    # x_sum = np.zeros(img_bit.shape[1])
    # for j in range(img_bit.shape[0]):
    #     x_sum += 255 - img_bit[j, :]
    # x_sum = x_sum / 255
    # print(x_sum)
    #
    # plt.subplot(121)
    # plt.imshow(img_bit, cmap='gray')
    # plt.subplot(122)
    # plt.plot(x_sum)
    # plt.show()



    cv2.imshow("source", img_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Check the valid pixels (black pixels) ratio in rectangule
def checkRectangle(rect, threshold):
    if rect is None or threshold == 0.0:
        return False
    area = rect.shape[0] * rect.shape[1]
    if area == 0.0:
        return False

    # white background (255)
    rect = 255.0 - rect
    valid = np.sum(np.array(rect) / 255)
    ratio = valid / area

    if ratio >= threshold:
        return True
    else:
        return False


# get center of rectangles (x, y, w, h)
def getCenterOfRectangles(rect):
    if rect is None:
        return None
    cx = rect[0] + int(rect[2]/2)
    cy = rect[1] + int(rect[3]/2)

    return (cx, cy)


def combineRectangles(rectangles, rect_list):
    if rectangles is None:
        return None
    if len(rect_list) == 1:
        return rectangles[rect_list[0]]

    new_rect_x = rectangles[rect_list[0]][0]
    new_rect_y = rectangles[rect_list[0]][1]
    new_rect_w = rectangles[rect_list[0]][2]
    new_rect_h = rectangles[rect_list[0]][3]

    for id in range(1, len(rect_list)):
        rect_x = rectangles[rect_list[id]][0]
        rect_y = rectangles[rect_list[id]][1]
        rect_w = rectangles[rect_list[id]][2]
        rect_h = rectangles[rect_list[id]][3]

        new_rect_x = min(new_rect_x, rect_x)
        new_rect_y = min(new_rect_y, rect_y)

        new_rect_w = max(new_rect_x+new_rect_w, rect_x+rect_w) - new_rect_x
        new_rect_h = max(new_rect_y+new_rect_h, rect_y+rect_h) - new_rect_y

    return new_rect_x, new_rect_y, new_rect_w, new_rect_h




if __name__ == '__main__':
    main()