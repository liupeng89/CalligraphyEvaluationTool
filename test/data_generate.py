# coding: utf-8
import os
import cv2
import numpy as np

from utils.Functions import getConnectedComponents

path = "/Users/liupeng/Documents/PythonProjects/九宫格修复版/九成宫修复版单字"

files = os.listdir(path)


for file in files:
    print("process:", file)
    if ".png" not in file:
        continue
    name = file.replace(".png", "")
    if not os.path.exists(os.path.join(path, name)):
        os.makedirs(os.path.join(path, name))

    if not os.path.exists(os.path.join(path, name, "original")):
        os.makedirs(os.path.join(path, name, "original"))
    if not os.path.exists(os.path.join(path, name, "char")):
        os.makedirs(os.path.join(path, name, "char"))
    if not os.path.exists(os.path.join(path, name, "strokes")):
        os.makedirs(os.path.join(path, name, "strokes"))

    os.system("cp %s %s" % (os.path.join(path, file), os.path.join(path, name, "original")))

    # processing
    img = cv2.imread(os.path.join(path, file), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    img = np.array(img)

    # moments
    im2, contours, hierarchy = cv2.findContours(img, 1, 2)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < 30 or h < 30:
            img[y: y+h, x: x+w] = 255

    img = cv2.resize(img, (int(img.shape[1]/2.), int(img.shape[0]/2.)))

    cv2.imwrite(os.path.join(path, name, "char", file), img)

    os.system("convert %s %s" % (os.path.join(path, name, "char", file), os.path.join(path, name, "char", name+".pnm")))
    os.system(
        "potrace -s -o %s %s" % (os.path.join(path, name, "char", name + ".svg"),
                                 os.path.join(path, name, "char", name + ".pnm")))




