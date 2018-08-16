"""
Aesthetic evaluation of Chinese calligraphy.

Input:    template image and strokes of this template image.

          templates/
                    ben/
                        char/
                        original/
                        strokes/
          targets/
                    ben/
                        char/
                        original/
                        strokes/

          target image and strokes of this target image.

The template and target images are pre-processed by denoising, contours smoothing
and stroke extraction, etc.

"""
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from utils.Functions import resizeImages, calculateCoverageRate, shiftImageWithMaxCR, getCenterOfGravity, \
    getConvexHullOfImage, calculateConvexHullArea, rotateImage, separateImageWithJiuGongGrid, \
    getSingleMaxBoundingBoxOfImage, separateImageWithElesticMesh


def aesthetic_evaluation(template_path, target_path):
    if template_path == "" or target_path == "":
        print("template path and target path are null!")
        return

    """
        Image pre-processing of template and target.
    """

    # 1. find template and target images and strokes
    template_char_path = template_path + "/char/"
    template_strokes_path = template_path + "/strokes/"

    target_char_path = target_path + "/char/"
    target_strokes_path = target_path + "/strokes/"

    # 2. load image from these image paths
    template_char_img = cv2.imread(template_char_path + "ben.png")
    target_char_img = cv2.imread(target_char_path + "ben.png")

    # 3. rgb imag to grayscale image
    template_char_img_gray = cv2.cvtColor(template_char_img, cv2.COLOR_RGB2GRAY)
    target_char_img_gray = cv2.cvtColor(target_char_img, cv2.COLOR_RGB2GRAY)

    # 4. Resize template and target images
    template_char_img_gray, target_char_img_gray = resizeImages(template_char_img_gray, target_char_img_gray)

    template_char_img_gray = np.array(template_char_img_gray, dtype=np.uint8)
    target_char_img_gray = np.array(target_char_img_gray, dtype=np.uint8)
    print("Resized shape:",template_char_img_gray.shape, template_char_img_gray.shape)

    # 5. Stroke extraction and save to fixed directory
    # 6. Load strokes of template and target images;
    template_strokes_img = []
    target_strokes_img = []

    for fl in os.listdir(template_strokes_path):
        if ".png" in fl:
            stroke = cv2.imread(template_strokes_path + "/" + fl)
            if len(stroke.shape) == 3:
                stroke = cv2.cvtColor(stroke, cv2.COLOR_RGB2GRAY)
            _, stroke = cv2.threshold(stroke, 127, 255, cv2.THRESH_BINARY)
            template_strokes_img.append(stroke)

    for fl in os.listdir(target_strokes_path):
        if ".png" in fl:
            stroke = cv2.imread(target_strokes_path + "/" + fl)
            if len(stroke.shape) == 3:
                stroke = cv2.cvtColor(stroke, cv2.COLOR_RGB2GRAY)
            _, stroke = cv2.threshold(stroke, 127, 255, cv2.THRESH_BINARY)
            target_strokes_img.append(stroke)
    print("template stroke num: %d , and target strokes num: %d" % (len(template_strokes_img), len(target_strokes_img)))

    # 2. Aesthetic evaluation of template and target.

    """
        1. Global features
    """

    # 1.1 max CR

    # shift the target image to get maximal CR
    target_char_img_gray = shiftImageWithMaxCR(template_char_img_gray, target_char_img_gray)
    max_cr = calculateCoverageRate(template_char_img_gray, target_char_img_gray)
    print("max cr: %0.3f" % max_cr)

    # 1.2 Center of gravity
    # template_cog = getCenterOfGravity(template_char_img_gray)
    # target_cog = getCenterOfGravity(target_char_img_gray)
    #
    # print("center of gravity: ", template_cog, target_cog)

    # 1.3 convex hull area
    # template_convex_hull = getConvexHullOfImage(template_char_img_gray)
    # target_convex_hull = getConvexHullOfImage(target_char_img_gray)
    #
    # template_convex_area = calculateConvexHullArea(template_convex_hull)
    # target_convex_area = calculateConvexHullArea(target_convex_hull)
    #
    # print("convex area:", template_convex_area, target_convex_area)
    #
    # template_valid_pixel_area = 0
    # target_valid_pixel_area = 0
    # # calculate the valid pixel of template and target images.
    # for y in range(template_char_img_gray.shape[0]):
    #     for x in range(template_char_img_gray.shape[1]):
    #         if template_char_img_gray[y][x] == 0.0:
    #             template_valid_pixel_area += 1
    #
    # for y in range(target_char_img_gray.shape[0]):
    #     for x in range(target_char_img_gray.shape[1]):
    #         if target_char_img_gray[y][x] == 0.0:
    #             target_valid_pixel_area += 1
    # print("valid pixel area:", template_valid_pixel_area, target_valid_pixel_area)
    #
    # template_convex_size_ratio = template_convex_area / (template_char_img_gray.shape[0] * template_char_img_gray.shape[1])
    # target_convex_size_ratio = target_convex_area / (target_char_img_gray.shape[0] * target_char_img_gray.shape[1])
    #
    # template_valid_convex_ratio = template_valid_pixel_area / template_convex_area
    # target_valid_convex_ratio = target_valid_pixel_area / target_convex_area
    #
    # print("convex size ratio:", template_convex_size_ratio, target_convex_size_ratio)
    # print("valid convex ratio:", template_valid_convex_ratio, target_valid_convex_ratio)

    # 1.4. histograms of template and target images at 0, 45, 90, 135 degree.

    # template_char_img_gray = np.array(template_char_img_gray, np.uint8)
    # target_char_img_gray = np.array(target_char_img_gray, np.uint8)
    #
    # # 0 degree
    # template_x_axis = np.zeros((template_char_img_gray.shape[1], 1))
    # template_y_axis = np.zeros((template_char_img_gray.shape[0], 1))
    # target_x_axis = np.zeros((target_char_img_gray.shape[1], 1))
    # target_y_axis = np.zeros((target_char_img_gray.shape[0], 1))
    #
    # for y in range(template_char_img_gray.shape[0]):
    #     for x in range(template_char_img_gray.shape[1]):
    #         if template_char_img_gray[y][x] == 0.0:
    #             template_x_axis[x] += 1
    #             template_y_axis[y] += 1
    # for y in range(target_char_img_gray.shape[0]):
    #     for x in range(target_char_img_gray.shape[1]):
    #         if target_char_img_gray[y][x] == 0.0:
    #             target_x_axis[x] += 1
    #             target_y_axis[y] += 1
    #
    # # mean
    # template_x_axis_mean = np.mean(template_x_axis)
    # template_y_axis_mean = np.mean(template_y_axis)
    # target_x_axis_mean = np.mean(target_x_axis)
    # target_y_axis_mean = np.mean(target_y_axis)
    #
    # template_x_axis_median = np.median(template_x_axis)
    # template_y_axis_median = np.median(template_y_axis)
    # target_x_axis_median = np.median(target_x_axis)
    # target_y_axis_median = np.median(target_y_axis)
    #
    # print("0 degree mean:", template_x_axis_mean, template_y_axis_mean, target_x_axis_mean, target_y_axis_mean)
    # print("0 degree median:", template_x_axis_median, template_y_axis_median, target_x_axis_median, target_y_axis_median)
    #
    # # display the template and target x-axis and y-axis
    # labels = list(range(template_char_img_gray.shape[0]))
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(labels, template_x_axis)
    # plt.title("template x axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(labels, template_y_axis)
    # plt.title("template y axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(labels, target_x_axis)
    # plt.title("target x axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(labels, target_y_axis)
    # plt.title("target y axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.show()
    #
    # # 45 degree
    # template_char_img_gray_45 = rotateImage(template_char_img_gray, 45)
    # target_char_img_gray_45 = rotateImage(target_char_img_gray, 45)
    #
    # template_x_axis = np.zeros((template_char_img_gray_45.shape[0], 1))
    # template_y_axis = np.zeros((template_char_img_gray_45.shape[1], 1))
    # target_x_axis = np.zeros((target_char_img_gray_45.shape[0], 1))
    # target_y_axis = np.zeros((target_char_img_gray_45.shape[1], 1))
    #
    # for y in range(template_char_img_gray_45.shape[0]):
    #     for x in range(template_char_img_gray_45.shape[1]):
    #         if template_char_img_gray_45[y][x] == 0.0:
    #             template_x_axis[x] += 1
    #             template_y_axis[y] += 1
    # for y in range(target_char_img_gray_45.shape[0]):
    #     for x in range(target_char_img_gray_45.shape[1]):
    #         if target_char_img_gray_45[y][x] == 0.0:
    #             target_x_axis[x] += 1
    #             target_y_axis[y] += 1
    #
    # # mean
    # template_x_axis_mean = np.mean(template_x_axis)
    # template_y_axis_mean = np.mean(template_y_axis)
    # target_x_axis_mean = np.mean(target_x_axis)
    # target_y_axis_mean = np.mean(target_y_axis)
    #
    # template_x_axis_median = np.median(template_x_axis)
    # template_y_axis_median = np.median(template_y_axis)
    # target_x_axis_median = np.median(target_x_axis)
    # target_y_axis_median = np.median(target_y_axis)
    #
    # print("45 degree mean:", template_x_axis_mean, template_y_axis_mean, target_x_axis_mean, target_y_axis_mean)
    # print("45 degree median:", template_x_axis_median, template_y_axis_median, target_x_axis_median,
    #           target_y_axis_median)
    #
    # # display the template and target x-axis and y-axis
    # labels = list(range(template_char_img_gray.shape[0]))
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(labels, template_x_axis)
    # plt.title("template x axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(labels, template_y_axis)
    # plt.title("template y axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(labels, target_x_axis)
    # plt.title("target x axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(labels, target_y_axis)
    # plt.title("target y axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.show()
    #
    #
    # # 90 degree
    # template_char_img_gray_90 = rotateImage(template_char_img_gray, 90)
    # target_char_img_gray_90 = rotateImage(target_char_img_gray, 90)
    #
    # template_x_axis = np.zeros((template_char_img_gray_90.shape[0], 1))
    # template_y_axis = np.zeros((template_char_img_gray_90.shape[1], 1))
    # target_x_axis = np.zeros((target_char_img_gray_90.shape[0], 1))
    # target_y_axis = np.zeros((target_char_img_gray_90.shape[1], 1))
    #
    # for y in range(template_char_img_gray_90.shape[0]):
    #     for x in range(template_char_img_gray_90.shape[1]):
    #         if template_char_img_gray_90[y][x] == 0.0:
    #             template_x_axis[x] += 1
    #             template_y_axis[y] += 1
    # for y in range(target_char_img_gray_90.shape[0]):
    #     for x in range(target_char_img_gray_90.shape[1]):
    #         if target_char_img_gray_90[y][x] == 0.0:
    #             target_x_axis[x] += 1
    #             target_y_axis[y] += 1
    #
    # # mean
    # template_x_axis_mean = np.mean(template_x_axis)
    # template_y_axis_mean = np.mean(template_y_axis)
    # target_x_axis_mean = np.mean(target_x_axis)
    # target_y_axis_mean = np.mean(target_y_axis)
    #
    # template_x_axis_median = np.median(template_x_axis)
    # template_y_axis_median = np.median(template_y_axis)
    # target_x_axis_median = np.median(target_x_axis)
    # target_y_axis_median = np.median(target_y_axis)
    #
    # print("90 degree mean:", template_x_axis_mean, template_y_axis_mean, target_x_axis_mean, target_y_axis_mean)
    # print("90 degree median:", template_x_axis_median, template_y_axis_median, target_x_axis_median,
    #           target_y_axis_median)
    #
    # # display the template and target x-axis and y-axis
    # labels = list(range(template_char_img_gray.shape[0]))
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(labels, template_x_axis)
    # plt.title("template x axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(labels, template_y_axis)
    # plt.title("template y axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(labels, target_x_axis)
    # plt.title("target x axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(labels, target_y_axis)
    # plt.title("target y axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.show()
    #
    # # 135 degree
    # template_char_img_gray_135 = rotateImage(template_char_img_gray, 135)
    # target_char_img_gray_135 = rotateImage(target_char_img_gray, 135)
    #
    # template_x_axis = np.zeros((template_char_img_gray_135.shape[0], 1))
    # template_y_axis = np.zeros((template_char_img_gray_135.shape[1], 1))
    # target_x_axis = np.zeros((target_char_img_gray_135.shape[0], 1))
    # target_y_axis = np.zeros((target_char_img_gray_135.shape[1], 1))
    #
    # for y in range(template_char_img_gray_135.shape[0]):
    #     for x in range(template_char_img_gray_135.shape[1]):
    #         if template_char_img_gray_135[y][x] == 0.0:
    #             template_x_axis[x] += 1
    #             template_y_axis[y] += 1
    # for y in range(target_char_img_gray_135.shape[0]):
    #     for x in range(target_char_img_gray_135.shape[1]):
    #         if target_char_img_gray_135[y][x] == 0.0:
    #             target_x_axis[x] += 1
    #             target_y_axis[y] += 1
    #
    # # mean
    # template_x_axis_mean = np.mean(template_x_axis)
    # template_y_axis_mean = np.mean(template_y_axis)
    # target_x_axis_mean = np.mean(target_x_axis)
    # target_y_axis_mean = np.mean(target_y_axis)
    #
    # template_x_axis_median = np.median(template_x_axis)
    # template_y_axis_median = np.median(template_y_axis)
    # target_x_axis_median = np.median(target_x_axis)
    # target_y_axis_median = np.median(target_y_axis)
    #
    # print("135 degree mean:", template_x_axis_mean, template_y_axis_mean, target_x_axis_mean, target_y_axis_mean)
    # print("135 degree median:", template_x_axis_median, template_y_axis_median, target_x_axis_median, target_y_axis_median)
    #
    # # display the template and target x-axis and y-axis
    # labels = list(range(template_char_img_gray.shape[0]))
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(labels, template_x_axis)
    # plt.title("template x axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(labels, template_y_axis)
    # plt.title("template y axis")
    # plt.ylabel("Number of valid pixels")
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(labels, target_x_axis)
    # plt.title("target x axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(labels, target_y_axis)
    # plt.title("target y axis")
    # plt.ylabel("Number of vaild pixels")
    #
    # plt.show()

    # 1.5. Statistics on Jiu-gong grid    1 | 2 | 3
    #                                   4 | 5 | 6
    #                                   7 | 8 | 9

    # mesh_shape = 3
    #
    # template_grids = separateImageWithElesticMesh(template_char_img_gray, mesh_shape)
    # target_grids = separateImageWithElesticMesh(target_char_img_gray, mesh_shape)
    #
    # # statistics of valid pixels in grids
    # template_grids_statis = []
    # target_grids_statis = []
    # for i in range(len(template_grids)):
    #     grid = template_grids[i]
    #     grid = 255 - grid
    #     num = np.sum(grid) / 255.
    #     template_grids_statis.append(num)
    # print(template_grids_statis)
    #
    # for i in range(len(target_grids)):
    #     grid = target_grids[i]
    #     grid = 255 - grid
    #     num = np.sum(grid) / 255.
    #     target_grids_statis.append(num)
    # print(target_grids_statis)

    # 1.6. Aspect ratio
    # _, _, temp_mini_box_w, temp_mini_box_h = getSingleMaxBoundingBoxOfImage(template_char_img_gray)
    # _, _, targ_mini_box_w, targ_mini_box_h = getSingleMaxBoundingBoxOfImage(target_char_img_gray)
    #
    # print(temp_mini_box_w, temp_mini_box_h, (temp_mini_box_w / temp_mini_box_h * 1.))
    # print(targ_mini_box_w, targ_mini_box_h, (targ_mini_box_w / targ_mini_box_h * 1.))

    # 1.7. elastic mesh effective statistics
    # mesh_shape = 9
    #
    # template_mesh_grids = separateImageWithElesticMesh(template_char_img_gray, mesh_shape)
    # target_mesh_grids = separateImageWithElesticMesh(target_char_img_gray, mesh_shape)
    # print(len(template_mesh_grids))
    # print(len(target_mesh_grids))
    #
    # # statistics effective area in grid of effective area of template and target images.
    # template_mesh_effective_area = []
    # target_mesh_effective_area = []
    # # template
    # for i in range(len(template_mesh_grids)):
    #     grid = template_mesh_grids[i]   # grid of template
    #     grid = 255 - grid               # inverse grid
    #     num = np.sum(grid) / 255.       # sum the effective area pixels of template
    #     template_mesh_effective_area.append(num)
    # print(template_mesh_effective_area)
    #
    # # target
    # for i in range(len(target_mesh_grids)):
    #     grid = target_mesh_grids[i]
    #     grid = 255 - grid
    #     num = np.sum(grid) / 255.
    #     target_mesh_effective_area.append(num)
    # print(target_mesh_effective_area)


    """
        2. Radical features
    """

    # 2.1 Radicals layout



    # 2.2 Effective area of radical.

    """
        3. Stroke features
        
    """
    if len(template_strokes_img) != len(target_strokes_img):
        print("Strokes number is different!!!")
    else:
        print("Stroke number is same!")

    # 3.1 Coverage rate
    # strokes_cr = []
    #
    # for i in range(len(template_strokes_img)):
    #     temp_stroke = template_strokes_img[i]
    #     targ_stroke = target_strokes_img[i]
    #     targ_stroke = shiftImageWithMaxCR(temp_stroke, targ_stroke)
    #
    #     cr_ = calculateCoverageRate(temp_stroke, targ_stroke)
    #
    #     strokes_cr.append(cr_)
    # print(strokes_cr)

    # 3.2 Genter of gravity
    # template_strokes_cog = []
    # target_strokes_cog = []
    #
    # for i in range(len(template_strokes_img)):
    #     stroke = template_strokes_img[i]
    #     stroke_cog = getCenterOfGravity(stroke)
    #     template_strokes_cog.append(stroke_cog)
    #
    # for i in range(len(target_strokes_img)):
    #     stroke = target_strokes_img[i]
    #     stroke_cog = getCenterOfGravity(stroke)
    #     target_strokes_cog.append(stroke_cog)
    #
    # print(template_strokes_cog)
    # print(target_strokes_cog)

    # 3.3 Convex hull area
    # template_strokes_convex = []
    # target_strokes_convex = []
    #
    # for i in range(len(template_strokes_img)):
    #     stroke = template_strokes_img[i]
    #     convex_ = getConvexHullOfImage(stroke)
    #     template_strokes_convex.append(convex_)
    #
    # for i in range(len(target_strokes_img)):
    #     stroke = target_strokes_img[i]
    #     convex_ = getConvexHullOfImage(stroke)
    #     target_strokes_convex.append(convex_)
    #
    # # calculate the convex hull area of template and target
    # template_strokes_convex_area = []
    # target_strokes_convex_area = []
    # for convex_ in template_strokes_convex:
    #     area = calculateConvexHullArea(convex_)
    #     template_strokes_convex_area.append(area)
    # for convex_ in target_strokes_convex:
    #     area = calculateConvexHullArea(convex_)
    #     target_strokes_convex_area.append(area)
    # print(template_strokes_convex_area)
    # print(target_strokes_convex_area)

    # 3.4 Begin, end and middle segmentation of stroke

    # 3.5 Angle of stroke: heng and shu
    # for i in range(len(template_strokes_img)):
    #     temp_stroke = template_strokes_img[i]
    #     targ_stroke = target_strokes_img[i]
    #
    #     # detect stroke is heng or shu
    #     _, _, temp_w, temp_h = getSingleMaxBoundingBoxOfImage(temp_stroke)
    #     _, _, targ_w, targ_h = getSingleMaxBoundingBoxOfImage(targ_stroke)
    #
    #     if temp_w > temp_h and temp_h / temp_w * 1. <= 0.5:
    #         print("heng")
    #
    #     elif temp_h > temp_w and temp_w / temp_h * 1. <= 0.2:
    #         print("shu")
































    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    temlplate_path = "test_images/templates/ben"
    target_path = "test_images/targets/ben"

    result = aesthetic_evaluation(temlplate_path, target_path)




if __name__ == '__main__':
    main()