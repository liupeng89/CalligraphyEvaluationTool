import cv2
import numpy as np

from utils.Functions import getSingleMaxBoundingBoxOfImage, resizeImages


def main():
    temp_path = "/Users/liupeng/Documents/PythonProjects/templates/templates/ben/char/ben.png"
    targ_path = "/Users/liupeng/Documents/PythonProjects/templates/templates_comparison/ben/char/ben.png"

    temp_img = cv2.imread(temp_path, 0)
    targ_img = cv2.imread(targ_path, 0)

    _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)
    _, targ_img = cv2.threshold(targ_img, 127, 255, cv2.THRESH_BINARY)

    # resize two images of template and target.
    temp_img, targ_img = resizeImages(temp_img, targ_img)

    temp_img = np.array(temp_img, dtype=np.uint8)
    targ_img = np.array(targ_img, dtype=np.uint8)

    # bounding box of template and target images
    temp_x, temp_y, temp_w, temp_h = getSingleMaxBoundingBoxOfImage(temp_img)
    targ_x, targ_y, targ_w, targ_h = getSingleMaxBoundingBoxOfImage(targ_img)

    temp_ct_x = temp_x + int(temp_w / 2.)
    temp_ct_y = temp_y + int(temp_h / 2.)

    targ_ct_x = targ_x + int(targ_w / 2.)
    targ_ct_y = targ_y + int(targ_h / 2.)

    # new square width
    square_width = max(temp_w, temp_h, targ_w, targ_h)

    # using new square to crop all effective area in template and target images
    # template image
    if temp_ct_x - int(square_width / 2.) <= 0:
        temp_x = 0
    else:
        temp_x = temp_ct_x - int(square_width / 2.)

    if temp_ct_y - int(square_width / 2,) <= 0:
        temp_y = 0
    else:
        temp_y = temp_ct_y - int(square_width / 2,)

    if temp_ct_x + int(square_width / 2.) >= temp_img.shape[1]:
        temp_w = temp_img.shape[1] - temp_x
    else:
        temp_w = square_width

    if temp_ct_y + int(square_width / 2.) >= temp_img.shape[0]:
        temp_h = temp_img.shape[0] - temp_y
    else:
        temp_h = square_width
    # target image
    if targ_ct_x - int(square_width / 2.) <= 0:
        targ_x = 0
    else:
        targ_x = targ_ct_x - int(square_width / 2.)

    if targ_ct_x - int(square_width / 2, ) <= 0:
        targ_y = 0
    else:
        targ_y = targ_ct_x - int(square_width / 2, )

    if targ_ct_x + int(square_width / 2.) >= targ_img.shape[1]:
        targ_w = targ_img.shape[1] - targ_x
    else:
        targ_w = square_width

    if targ_ct_x + int(square_width / 2.) >= targ_img.shape[0]:
        targ_h = targ_img.shape[0] - targ_y
    else:
        targ_h = square_width

    # crop effective areas of the template and target images
    temp_reg = temp_img[temp_y:temp_y+temp_h, temp_x:temp_x+temp_w]
    targ_reg = targ_img[targ_y:targ_y+targ_h, targ_x:targ_x+targ_w]

    shape_similarity = calculateShapeSimilarity(temp_reg, targ_reg)
    simi = calculateShapeSimilarity(temp_reg, temp_reg)

    print("Shape similarity: %f" % shape_similarity)
    print("Same image similarity: %f" % simi)

    cv2.imshow("temp", temp_reg)
    cv2.imshow("targ", targ_reg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculateShapeSimilarity(square1, square2):
    if square1 is None or square2 is None:
        return 0.0
    # same shape of two squares
    if square1.shape != square2.shape:
        print("Two image shape are different!")
        return 0.0
    #

    h = square1.shape[0]; w = square1.shape[1]

    square1_mat = np.zeros_like(square1)
    square2_mat = np.zeros_like(square2)

    for y in range(h):
        for x in range(w):
            if square1[y][x] == 0.0:
                square1_mat[y][x] = 1.
            if square2[y][x] == 0.0:
                square2_mat[y][x] = 1.

    #
    numerator = np.sum(square1_mat * square2_mat)
    abs_ = np.sum(np.abs(square1_mat - square2_mat))

    similarity = numerator / (numerator + abs_)

    return similarity


if __name__ == '__main__':
    main()
    # a = [[1,1,1], [0,0,0], [1,0,1]]
    # b = [[0,1,0],[1,1,0],[0,0,1]]
    #
    # a = np.array(a)
    # b = np.array(b)
    #
    # si = calculateShapeSimilarity(a, b)
    # print(si)