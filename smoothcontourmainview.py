import sys
import math
import cv2
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from smoothcontourmainwindow import Ui_MainWindow
from utils.Functions import getContourOfImage, getNumberOfValidPixels, sortPointsOnContourOfImage, \
                            fitCurve, draw_cubic_bezier


class SmoothContourMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(SmoothContourMainWindow, self).__init__()
        self.setupUi(self)

        # original image scene
        self.original_img_scene = QGraphicsScene()
        self.original_img_gview.setScene(self.original_img_scene)

        # original contour scene
        self.original_contour_scene = QGraphicsScene()
        self.original_contour_gview.setScene(self.original_contour_scene)

        # smooth image scene
        self.smooth_img_scene = QGraphicsScene()
        self.smooth_img_gview.setScene(self.smooth_img_scene)

        # smooth contour scene
        self.smooth_contour_scene = QGraphicsScene()
        self.smooth_contour_gview.setScene(self.smooth_contour_scene)

        # add listen adapter
        self.import_btn.clicked.connect(self.importBtn)
        self.smooth_btn.clicked.connect(self.smoothBtn)
        self.save_btn.clicked.connect(self.saveBtn)
        self.exit_btn.clicked.connect(self.exitBtn)

        # image path
        self.image_path = ""
        self.image_name = ""

        self.image_gray = None
        self.image_contour = None
        self.image_contour_rgb = None

    def importBtn(self):
        """
            Impprt image file
        :return:
        """
        print("Import button clicked!")
        # clean all views
        self.original_img_scene.clear()
        self.original_contour_scene.clear()
        self.smooth_img_scene.clear()
        self.smooth_contour_scene.clear()

        # open image file
        filename, _ = QFileDialog.getOpenFileName(None, "Open file", QDir.currentPath())
        if filename:
            # image file path
            self.image_path = filename
            self.image_name = os.path.splitext(os.path.basename(filename))[0]

            qimage = QImage(filename)
            if qimage.isNull():
                QMessageBox.information(self, "Image viewer", "Cannot load %s." % filename)
                return

            # grayscale image
            img_ = cv2.imread(filename, 0)
            _, img_ = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY)

            # grayscale image
            self.image_gray = img_.copy()

            # contour image
            contour = getContourOfImage(img_)
            contour = np.array(contour, dtype=np.uint8)
            self.image_contour = contour.copy()
            contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
            self.image_contour_rgb = contour_rgb.copy()

            # show on the views
            image_pix = QPixmap.fromImage(qimage)
            self.original_img_scene.addPixmap(image_pix)

            qcontour = QImage(contour.data, contour.shape[1], contour.shape[0], contour.shape[1], \
                                 QImage.Format_Indexed8)
            contour_pix = QPixmap.fromImage(qcontour)
            self.original_contour_scene.addPixmap(contour_pix)

            self.original_img_scene.update()
            self.original_contour_scene.update()

    def smoothBtn(self):
        """
        Smooth contour.
        :return:
        """
        self.smooth_img_scene.clear()
        self.smooth_contour_scene.clear()
        print("Smooth button clicked")
        numOfFeaturePoints = 0
        maxError = 0.

        try:
            numOfFeaturePoints = int(self.featurepts_ledit.text())
            maxError = int(self.maxerror_ledit.text())
        except:
            print("Input error")
            return
        # smooth
        contour_gray = self.image_contour.copy()
        contour_rgb = self.image_contour_rgb.copy()

        # fix breaking points on the contour
        break_points = []
        for y in range(1, contour_gray.shape[0] - 1):
            for x in range(1, contour_gray.shape[1] - 1):
                if contour_gray[y][x] == 0.0:
                    num_ = getNumberOfValidPixels(contour_gray, x, y)
                    if num_ == 1:
                        print((x, y))
                        break_points.append((x, y))
        if len(break_points) != 0:
            contour_gray = cv2.line(contour_gray, break_points[0], break_points[1], color=0, thickness=1)

        # sort the points on the contour
        contour_points_ordered = sortPointsOnContourOfImage(contour_gray)

        # get the feature points on contour
        corners = cv2.goodFeaturesToTrack(contour_gray, numOfFeaturePoints, 0.01, 10)
        corners = np.int0(corners)

        # keep the corner points should be all points on the contour
        corner_points_ = []
        for i in corners:
            MAX_DIST = 10000
            x, y = i.ravel()
            pt_ = None
            if (x, y - 1) in contour_points_ordered:
                pt_ = (x, y - 1)
            elif (x + 1, y - 1) in contour_points_ordered:
                pt_ = (x + 1, y - 1)
            elif (x + 1, y) in contour_points_ordered:
                pt_ = (x + 1, y)
            elif (x + 1, y + 1) in contour_points_ordered:
                pt_ = (x + 1, y + 1)
            elif (x, y + 1) in contour_points_ordered:
                pt_ = (x, y + 1)
            elif (x - 1, y + 1) in contour_points_ordered:
                pt_ = (x - 1, y + 1)
            elif (x - 1, y) in contour_points_ordered:
                pt_ = (x - 1, y)
            elif (x - 1, y - 1) in contour_points_ordered:
                pt_ = (x - 1, y - 1)
            else:
                # find the nearest point on the contour
                minx = 0
                miny = 0
                for cp in contour_points_ordered:
                    dist = math.sqrt((x - cp[0]) ** 2 + (y - cp[1]) ** 2)
                    if dist < MAX_DIST:
                        MAX_DIST = dist
                        minx = cp[0]
                        miny = cp[1]
                pt_ = (minx, miny)
            corner_points_.append(pt_)
        # sorted the corner points with clockwise direction
        corner_points = []
        index = 0
        for pt in contour_points_ordered:
            if pt in corner_points_:
                corner_points.append(pt)

        # contour segemetations based on the corner points
        contour_lines = []
        for id in range(len(corner_points)):
            start_point = corner_points[id]
            end_point = start_point
            if id == len(corner_points) - 1:
                end_point = corner_points[0]
            else:
                end_point = corner_points[id + 1]
            # contour segmentation
            contour_segmentation = []
            start_index = contour_points_ordered.index(start_point)
            end_index = contour_points_ordered.index(end_point)

            if start_index <= end_index:
                # normal index
                contour_segmentation = contour_points_ordered[start_index: end_index + 1]
            else:
                # end is at
                contour_segmentation = contour_points_ordered[start_index: len(contour_points_ordered)] + \
                                       contour_points_ordered[0: end_index + 1]
            contour_lines.append(contour_segmentation)

        # using different color to display the contour segmentations
        for id in range(len(contour_lines)):
            if id % 3 == 0:
                # red lines
                for pt in contour_lines[id]:
                    contour_rgb[pt[1]][pt[0]] = (0, 0, 255)
            elif id % 3 == 1:
                # blue line
                for pt in contour_lines[id]:
                    contour_rgb[pt[1]][pt[0]] = (255, 0, 0)
            elif id % 3 == 2:
                # green line
                for pt in contour_lines[id]:
                    contour_rgb[pt[1]][pt[0]] = (0, 255, 0)
        contour_rgb = np.array(contour_rgb)
        print(contour_rgb.shape)
        qoriginal_contour_rgb = rgb2qimage(contour_rgb)
        # qoriginal_contour_rgb = QImage(contour_rgb.data, contour_rgb.shape[1], contour_rgb.shape[0], contour_rgb.shape[1]*3, \
        #                   QImage.Format_RGB32)
        qorignal_contour_rgb_pix = QPixmap.fromImage(qoriginal_contour_rgb)
        self.smooth_img_scene.addPixmap(qorignal_contour_rgb_pix)
        self.smooth_img_scene.update()

        # smooth segmentations
        smoothed_contour_points = []
        for id in range(len(contour_lines)):

            # smooth contour segmentation
            li_points = np.array(contour_lines[id])

            beziers = fitCurve(li_points, maxError=30)
            print("len bezier: %d" % len(beziers))
            # # print(beziers)
            for bez in beziers:
                print(len(bez))
                bezier_points = draw_cubic_bezier(bez[0], bez[1], bez[2], bez[3])
                smoothed_contour_points += bezier_points

        # fill the smooth contour region
        smoothed_contour_points = np.array([smoothed_contour_points], "int32")
        fill_smooth_contour = np.ones(self.image_gray.shape) * 255
        fill_smooth_contour = np.array(fill_smooth_contour, dtype=np.uint8)
        fill_smooth_contour = cv2.fillPoly(fill_smooth_contour, smoothed_contour_points, 0)

        qfill_smooth_contour = QImage(fill_smooth_contour.data, fill_smooth_contour.shape[1], fill_smooth_contour.shape[0], fill_smooth_contour.shape[1], \
                          QImage.Format_Indexed8)
        qfill_smooth_contour_pix = QPixmap.fromImage(qfill_smooth_contour)
        self.smooth_contour_scene.addPixmap(qfill_smooth_contour_pix)
        self.smooth_contour_scene.update()



    def saveBtn(self):
        """
        Save smooth contour images
        :return:
        """
        print("Save button clicked")

    def exitBtn(self):
        """
        Exit button clicked.
        :return:
        """
        qApp = QApplication.instance()
        sys.exit(qApp.exec_())


def rgb2qimage(rgb):
    """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError(
            "rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape

    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[..., 0] = rgb[..., 2]
    bgra[..., 1] = rgb[..., 1]
    bgra[..., 2] = rgb[..., 0]
    if rgb.shape[2] == 3:
        bgra[..., 3].fill(255)
        fmt = QImage.Format_RGB32
    else:
        bgra[..., 3] = rgb[..., 3]
        fmt = QImage.Format_ARGB32

    result = QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra
    return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = SmoothContourMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())