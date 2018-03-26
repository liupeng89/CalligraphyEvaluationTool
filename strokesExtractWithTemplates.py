import os
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from utils.Functions import getSingleMaxBoundingBoxOfImage, calculateCoverageRate


def main():

    base_path = "../templates/templates/ben/strokes"

    source_path = "../templates/templates/ben/char/ben.png"

    source_img = cv2.imread(source_path, 0)
    _, source_img = cv2.threshold(source_img, 127, 255, cv2.THRESH_BINARY)

    stroke_windows = getStrokesListFromTemplates(base_path)
    print("strokes len: %d" % len(stroke_windows))

    # target image
    target_path = "../templates/templates_comparison/ben/char/ben.png"
    target_img = cv2.imread(target_path, 0)
    _, target_img = cv2.threshold(target_img, 127, 255, cv2.THRESH_BINARY)
    print(target_img.shape)
    # resize target image
    new_tg_h = max(source_img.shape[0], target_img.shape[0])
    new_tg_w = max(source_img.shape[1], target_img.shape[1])

    target_img = cv2.resize(target_img, (int(new_tg_h*0.2), int(new_tg_w*0.2)))


    print(target_img.shape)
    th = target_img.shape[0]; tw = target_img.shape[1]


    # for stroke_window in stroke_windows:
    #     cv2.imshow("stroke", stroke_window)
    for id, stroke_window in enumerate(stroke_windows):
        cv2.imshow("stroke_"+str(id), stroke_window)

    # slide the stroke window on the target image
    stroke_list = []
    for stroke_window in stroke_windows:
        # resize stroke window * 0.2
        stroke_window = cv2.resize(stroke_window, (int(stroke_window.shape[0] * 0.2), int(stroke_window.shape[1]*0.2)))

        print("stroke window size: (%d, %d)" % stroke_window.shape)
        sh = stroke_window.shape[0]; sw = stroke_window.shape[1]

        min_diff = 1000000
        max_ssim = 0.0
        max_cr = -10000000.0
        min_x = 0; min_y = 0

        for y in range(th-sh):
            for x in range(tw-sw):
                target_window = target_img[y: y+sh, x: x+sw]
                print(target_window.shape)
                # diff = np.sum((np.abs(target_window - stroke_window))/255)
                # diff_ssim = ssim(target_window, stroke_window)
                diff_cr = calculateCoverageRate(stroke_window, target_window)
                print("diff cr: %f" % diff_cr)

                if diff_cr > max_cr:
                    min_x = x
                    min_y = y
                    max_cr = diff_cr

                # if diff < min_diff:
                #     min_x = x
                #     min_y = y
                #     min_diff = diff

        print("min: (%d, %d)" % (min_x, min_y))
        print("max cr: %f" % max_cr)

        stroke_list.append((min_x, min_y, sw, sh))
        break



    print(len(stroke_list))

    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)

    for stroke in stroke_list:
        target_img_rgb[stroke[0]: stroke[0]+stroke[2], stroke[1]: stroke[1]+stroke[3]] = (255, 0, 0)


    cv2.imshow("rgb", target_img_rgb)
    # cv2.imshow("stroke", stroke_windows[0])


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getStrokesListFromTemplates(templates_path):
    """
        Obtain strokes from templates files.
    :param templates_path:
    :return:
    """
    stroke_windows = []
    if templates_path == "" or not os.path.exists(templates_path):
        return

    templates_files = [f for f in os.listdir(templates_path) if '.png' in f]
    print(templates_files)

    for temp_file in templates_files:
        temp_path = templates_path + "/" + temp_file
        temp_img = cv2.imread(temp_path, 0)
        # scale image
        # temp_img = cv2.resize(temp_img, (int(temp_img.shape[0]), int(temp_img.shape[1])))

        _, temp_img = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)
        print(temp_img.shape)

        # obtain the stroke window
        x, y, w, h = getSingleMaxBoundingBoxOfImage(temp_img)

        stroke_window = temp_img[y:y+h, x:x+w]
        print(stroke_window.shape)
        #
        # cv2.imshow(temp_path, stroke_window)
        stroke_windows.append(stroke_window)

    return stroke_windows








if __name__ == '__main__':
    main()

    # base_path = "../templates/templates/ben/strokes"
    # stroke_windows = getStrokesListFromTemplates(base_path)
    # print("strokes len: %d" % len(stroke_windows))
    #
    # for id, stroke_window in enumerate(stroke_windows):
    #     cv2.imshow("stroke_"+str(id), stroke_window)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



