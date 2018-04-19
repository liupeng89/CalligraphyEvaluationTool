import sys
import math
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from calligraphycomparisonmainwindow import Ui_MainWindow
from utils.Functions import coverTwoImages, shiftImageWithMaxCR, addIntersectedFig, addSquaredFig, resizeImages, \
                            calculateCoverageRate, rgb2qimage, getCenterOfGravity, getConvexHullOfImage, \
                            calculatePolygonArea, calculateValidPixelsArea, getSingleMaxBoundingBoxOfImage


class CalligraphyComparisonGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(CalligraphyComparisonGUI, self).__init__()
        self.setupUi(self)

        self.template_image_name = ""
        self.template_image_path = ""
        self.template_image_radical_path = ""
        self.template_image_radicals = []
        self.template_image_stroke_path = ""
        self.template_image_strokes = []
        self.template_image_strokes_name = []

        self.target_image_name = ""
        self.target_image_path = ""
        self.target_image_radical_path = ""
        self.target_image_radicals = []
        self.target_image_stroke_path = ""
        self.target_image_strokes = []
        self.target_image_strokes_name = []

        self.template_image_rgb = None
        self.template_image_gray = None
        self.template_image_gray_no_resize = None

        self.target_image_rgb = None
        self.target_image_gray = None
        self.target_image_gray_no_resize = None

        self.original_image_rgb = None
        self.target_image_rgb = None

        self.original_image_gray = None
        self.target_image_gray = None

        # scene
        self.template_scene = QGraphicsScene()
        self.template_scene.setBackgroundBrush(Qt.gray)
        self.template_gview.setScene(self.template_scene)
        self.template_scene.setSceneRect(QRectF())
        self.template_gview.fitInView(self.template_scene.sceneRect(), Qt.KeepAspectRatio)

        self.target_scene = QGraphicsScene()
        self.target_scene.setBackgroundBrush(Qt.gray)
        self.target_gview.setScene(self.target_scene)

        self.cover_scene = QGraphicsScene()
        self.cover_scene.setBackgroundBrush(Qt.gray)
        self.cover_gview.setScene(self.cover_scene)

        self.template_cog_scene = QGraphicsScene()
        self.template_cog_scene.setBackgroundBrush(Qt.gray)
        self.cog_template_gview.setScene(self.template_cog_scene)

        self.target_cog_scene = QGraphicsScene()
        self.target_cog_scene.setBackgroundBrush(Qt.gray)
        self.cog_target_gview.setScene(self.target_cog_scene)

        self.cog_comparison_scene = QGraphicsScene()
        self.cog_comparison_scene.setBackgroundBrush(Qt.gray)
        self.cog_comparison_gview.setScene(self.cog_comparison_scene)

        # convex hull scene
        self.template_ch_scene = QGraphicsScene()
        self.template_ch_scene.setBackgroundBrush(Qt.gray)
        self.template_ch_gview.setScene(self.template_ch_scene)

        self.target_ch_scene = QGraphicsScene()
        self.target_ch_scene.setBackgroundBrush(Qt.gray)
        self.target_ch_gview.setScene(self.target_ch_scene)

        # plot of project
        self.main_widget = QWidget(self)

        self.temp_x_canvas = MplCanvas(self.main_widget)
        self.temp_y_canvas = MplCanvas(self.main_widget)

        self.targ_x_canvas = MplCanvas(self.main_widget)
        self.targ_y_canvas = MplCanvas(self.main_widget)

        self.temp_x_glayout.addWidget(self.temp_x_canvas)
        self.temp_y_glayout.addWidget(self.temp_y_canvas)
        self.targ_x_glayout.addWidget(self.targ_x_canvas)
        self.targ_y_glayout.addWidget(self.targ_y_canvas)

        self.template_strokes_layout_scene = QGraphicsScene()
        self.template_strokes_layout_scene.setBackgroundBrush(Qt.gray)
        self.template_strokes_layout_gview.setScene(self.template_strokes_layout_scene)

        self.target_strokes_layout_scene = QGraphicsScene()
        self.target_strokes_layout_scene.setBackgroundBrush(Qt.gray)
        self.target_strokes_layout_gview.setScene(self.target_strokes_layout_scene)

        # Strokes coverage rate
        self.strokes_cr_slm = QStringListModel()
        self.strokes_cr_slm.setStringList(self.template_image_strokes_name)
        self.strokes_cr_listview.setModel(self.strokes_cr_slm)
        self.strokes_cr_listview.clicked.connect(self.strokes_item_selected)

        self.strokes_cr_scene = QGraphicsScene()
        self.strokes_cr_scene.setBackgroundBrush(Qt.gray)
        self.strokes_cr_gview.setScene(self.strokes_cr_scene)

        self.stroke_selected_name = ""
        self.template_stroke_selected_gray = None
        self.target_stroke_selected_gray = None

        # back the pixmap object
        self.whole_cr_pix = None
        self.whole_temp_goc_pix = None
        self.whole_targ_goc_pix = None
        self.whole_combine_goc_pix = None
        self.whole_temp_ch_pix = None
        self.whole_targ_ch_pix = None

        self.radical_temp_layout_pix = None
        self.radical_targ_layout_pix = None

        self.stroke_cr_pix = None

        # add listener
        self.open_template_btn.clicked.connect(self.openTemplatesBtn)
        self.open_target_btn.clicked.connect(self.openTargetsBtn)
        self.cr_calculate_btn.clicked.connect(self.calculateCRBtn)
        self.exit_btn.clicked.connect(self.exitBtn)

        self.cog_btn.clicked.connect(self.centerOfGravityBtn)
        self.convexhull_btn.clicked.connect(self.convexHullBtn)
        self.project_btn.clicked.connect(self.projectBtn)
        self.stroke_layout_btn.clicked.connect(self.storkeLayoutBtn)

        self.whole_main_tabwidget.currentChanged.connect(self.wholeMainTabWidgetOnChange)
        self.comparison_main_tab.currentChanged.connect(self.radicalMainTabWidgetOnChange)
        self.stroke_main_tabwidget.currentChanged.connect(self.strokeMainTabWidgetOnChange)

    def openTemplatesBtn(self):
        """
        Open template button.
        :return:
        """
        print("Open templates image button clicked!")

        self.template_scene.clear(); self.cover_scene.clear()

        filename, _ = QFileDialog.getOpenFileName(None, "Open template files", QDir.currentPath())
        if filename:
            # image file path and name
            self.template_image_path = filename
            self.template_image_name = os.path.splitext(os.path.basename(filename))[0]

            qimage = QImage(filename)
            if qimage.isNull():
                QMessageBox.information(self, "Template image viewer", "Cannot load %s." % filename)
                return

            # original rgb images of template
            img_rgb = cv2.imread(filename)
            self.template_image_rgb = img_rgb.copy()

            parent_path = str(os.path.split(os.path.dirname(filename))[0])
            gray_path = parent_path + "/char/" + self.template_image_name + ".png"

            try:
                img_gray = cv2.imread(gray_path, 0)
            except:
                QMessageBox.information(self, "Template gray image", "Cannot load %s grayscale image." % filename)
                return

            strokes_path = parent_path + "/strokes/"
            self.template_image_stroke_path = strokes_path
            self.template_image_strokes_name = []
            for fl in os.listdir(strokes_path):
                if ".png" in fl:
                    self.template_image_strokes_name.append(fl)

            # stroke comparison cr listview
            self.strokes_cr_slm.setStringList(self.template_image_strokes_name)

            _, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            self.template_image_gray = img_gray.copy()
            self.template_image_gray_no_resize = img_gray.copy()

            # show original rgb image of template
            self.template_image_pix = QPixmap.fromImage(qimage)
            self.template_scene.addPixmap(self.template_image_pix)
            # fit image in gview with fitting size
            self.template_scene.setSceneRect(QRectF())
            self.template_gview.fitInView(self.template_scene.sceneRect(), Qt.KeepAspectRatio)
            self.template_scene.update()

            self.statusbar.showMessage("Open template file: %s successed!" % self.template_image_name)

            del qimage, img_gray, img_rgb

    def openTargetsBtn(self):
        """
        Open Target image button.
        :return:
        """
        print("Open target image button clicked!")

        self.target_scene.clear()
        self.cover_scene.clear()

        filename, _ = QFileDialog.getOpenFileName(None, "Open target files", QDir.currentPath())
        if filename:
            # image file path and name
            self.target_image_path = filename
            self.target_image_name = os.path.splitext(os.path.basename(filename))[0]

            qimage = QImage(filename)
            if qimage.isNull():
                QMessageBox.information(self, "Target image viewer", "Cannot load %s." % filename)
                return

            # original rgb images of template
            img_rgb = cv2.imread(filename)
            self.target_image_rgb = img_rgb.copy()

            parent_path = str(os.path.split(os.path.dirname(filename))[0])

            gray_path = parent_path + "/char/" + self.target_image_name + ".png"
            try:

                img_gray = cv2.imread(gray_path, 0)
            except:
                QMessageBox.information(self, "Template gray image", "Cannot load %s grayscale image." % filename)
                return

            strokes_path = parent_path + "/strokes/"
            self.target_image_stroke_path = strokes_path
            self.target_image_strokes_name = []
            for fl in os.listdir(strokes_path):
                if ".png" in fl:
                    self.target_image_strokes_name.append(fl)

            _, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            self.target_image_gray_no_resize = img_gray.copy()

            # resize the tempate grayscale image based on the target grayscale image size
            if self.template_image_gray is None:
                QMessageBox.information(self, "Open target image", "Template image should not be none!")
                return
            template_image_gray, target_image_gray = resizeImages(self.template_image_gray, img_gray)

            # update template and target grayscale image
            template_image_gray = np.array(template_image_gray, dtype=np.uint8)
            target_image_gray = np.array(target_image_gray, dtype=np.uint8)
            self.template_image_gray = template_image_gray.copy()
            self.target_image_gray = target_image_gray.copy()
            print(self.template_image_gray.shape)
            print(self.target_image_gray.shape)

            # show original rgb image of template
            target_image_pix = QPixmap.fromImage(qimage)
            self.target_scene.addPixmap(target_image_pix)
            self.target_scene.setSceneRect(QRectF())
            self.target_gview.fitInView(self.target_scene.sceneRect(), Qt.KeepAspectRatio)
            self.target_scene.update()

            self.statusbar.showMessage("Open template file: %s successed!" % self.target_image_name)

            del qimage, img_rgb, img_gray,target_image_pix, template_image_gray, target_image_gray

    def calculateCRBtn(self):
        """
        Max coverage rate.
        :return:
        """
        print("Calculate Max cr button clicked!")
        self.cover_scene.clear()
        self.whole_cr_pix = None

        if self.template_image_gray is None or self.target_image_gray is None:
            QMessageBox.information(self, "Grayscale image", "Grayscale image is None!")
            return
        # grayscalue
        template_gray = self.template_image_gray.copy()
        target_gray = self.target_image_gray.copy()

        new_target_gray = shiftImageWithMaxCR(template_gray, target_gray)
        # new_target_gray = target_gray

        cover_image_rgb = coverTwoImages(template_gray, new_target_gray)

        cr = calculateCoverageRate(template_gray, new_target_gray)
        print("cr: %6.2f" % cr)

        self.maxcr_label.setText("%4.3f" % cr)

        cover_image_rgb = np.array(cover_image_rgb)
        qimg = rgb2qimage(cover_image_rgb)
        image_pix = QPixmap.fromImage(qimg)

        self.cover_scene.addPixmap(image_pix)
        self.cover_scene.setSceneRect(QRectF())
        self.cover_gview.fitInView(self.cover_scene.sceneRect(), Qt.KeepAspectRatio)
        self.cover_scene.update()
        self.whole_cr_pix = image_pix.copy()
        self.statusbar.showMessage("Cover images successed!")

        del template_gray, target_gray, new_target_gray, cover_image_rgb, image_pix

    def centerOfGravityBtn(self):
        print("Cog button clicked!")

        self.template_cog_scene.clear()
        self.target_cog_scene.clear()
        self.cog_comparison_scene.clear()
        self.whole_temp_goc_pix = None
        self.whole_targ_goc_pix = None
        self.whole_combine_goc_pix = None

        if self.template_image_gray is None or self.target_image_gray is None:
            QMessageBox.information(self, "Grayscale image", "Grayscale image is None!")
            return
        # grayscale images
        template_img_gray = self.template_image_gray.copy()
        target_img_gray = self.target_image_gray.copy()

        template_img_gray = np.array(template_img_gray, dtype=np.uint8)
        target_img_gray = np.array(target_img_gray, dtype=np.uint8)

        template_img_rgb = cv2.cvtColor(template_img_gray, cv2.COLOR_GRAY2RGB)
        target_img_rgb = cv2.cvtColor(target_img_gray, cv2.COLOR_GRAY2RGB)

        # calcuate the center of gravity
        temp_cog_x, temp_cog_y = getCenterOfGravity(template_img_gray)
        tar_cog_x, tar_cog_y = getCenterOfGravity(target_img_gray)

        template_img_rgb = cv2.circle(template_img_rgb, (temp_cog_y, temp_cog_x), 4, (0, 0, 255), -1)
        target_img_rgb = cv2.circle(target_img_rgb, (tar_cog_y, tar_cog_x), 4, (255, 0, 0), -1)

        # display template rgb image with center of gravity
        template_img_rgb = np.array(template_img_rgb)
        temp_qimg = rgb2qimage(template_img_rgb)
        temp_pix = QPixmap.fromImage(temp_qimg)
        self.template_cog_scene.addPixmap(temp_pix)
        self.template_cog_scene.setSceneRect(QRectF())
        self.cog_template_gview.fitInView(self.template_cog_scene.sceneRect(), Qt.KeepAspectRatio)
        self.template_cog_scene.update()
        self.whole_temp_goc_pix = temp_pix.copy()

        # display target rgb image with center of gravity
        target_img_rgb = np.array(target_img_rgb)
        target_qimg = rgb2qimage(target_img_rgb)
        target_pix = QPixmap.fromImage(target_qimg)
        self.target_cog_scene.addPixmap(target_pix)
        self.target_cog_scene.setSceneRect(QRectF())
        self.cog_target_gview.fitInView(self.target_cog_scene.sceneRect(), Qt.KeepAspectRatio)
        self.target_cog_scene.update()
        self.whole_targ_goc_pix = target_pix.copy()

        # combine two center of gravity between two images.
        print(template_img_gray.shape)
        combine_rgb = np.ones_like(template_img_gray) * 255
        combine_rgb = np.array(combine_rgb, dtype=np.uint8)
        print(combine_rgb.shape)
        combine_rgb = cv2.cvtColor(combine_rgb, cv2.COLOR_GRAY2RGB)

        # template image with red cog
        combine_rgb = cv2.circle(combine_rgb, (temp_cog_y, temp_cog_x), 4, (0, 0, 255), -1)
        # target image with blue cog
        combine_rgb = cv2.circle(combine_rgb, (tar_cog_y, tar_cog_x), 4, (255, 0, 0), -1)

        combine_rgb = np.array(combine_rgb)
        combine_qimg = rgb2qimage(combine_rgb)
        combine_pix = QPixmap.fromImage(combine_qimg)

        self.cog_comparison_scene.addPixmap(combine_pix)
        self.cog_comparison_scene.setSceneRect(QRectF())
        self.cog_comparison_gview.fitInView(self.cog_comparison_scene.sceneRect(), Qt.KeepAspectRatio)
        self.cog_comparison_scene.update()
        self.whole_combine_goc_pix = combine_pix.copy()

        # update cog labels
        self.template_cog_label.setText("(%d, %d)" % (temp_cog_x, temp_cog_y))
        self.target_cog_label.setText("(%d, %d)" % (tar_cog_x, tar_cog_y))

        self.statusbar.showMessage("Get center of gravity successed!")

        del template_img_gray, target_img_gray, template_img_rgb, target_img_rgb, temp_qimg, temp_pix, target_pix, \
            target_qimg, combine_rgb, combine_qimg, combine_pix

    def convexHullBtn(self):
        print("Convex hull button clicked!")

        self.template_ch_scene.clear()
        self.target_ch_scene.clear()
        self.whole_temp_ch_pix = None
        self.whole_targ_ch_pix = None

        if self.template_image_gray is None or self.target_image_gray is None:
            QMessageBox.information(self, "Grayscale image", "Grayscale image is None!")
            return

        # convex hull
        template_img_gray = self.template_image_gray.copy()
        target_img_gray = self.target_image_gray.copy()

        template_img_gray = np.array(template_img_gray, dtype=np.uint8)
        target_img_gray = np.array(target_img_gray, dtype=np.uint8)
        print(template_img_gray.shape)

        template_l = getConvexHullOfImage(template_img_gray)
        target_l = getConvexHullOfImage(target_img_gray)

        template_img_rgb = cv2.cvtColor(template_img_gray, cv2.COLOR_GRAY2RGB)
        target_img_rgb = cv2.cvtColor(target_img_gray, cv2.COLOR_GRAY2RGB)

        # add convex hull to template RGB image
        for idx in range(len(template_l)):
            if idx+1 == len(template_l):
                template_img_rgb = cv2.line(template_img_rgb, (template_l[idx][0], template_l[idx][1]), \
                                            (template_l[0][0], template_l[0][1]), (0, 0, 255), 2)
            else:
                template_img_rgb = cv2.line(template_img_rgb, (template_l[idx][0], template_l[idx][1]), \
                                            (template_l[idx+1][0], template_l[idx+1][1]), (0, 0, 255), 2)

        # add convex hull to target RGB image
        for idx in range(len(target_l)):
            if idx+1 == len(target_l):
                target_img_rgb = cv2.line(target_img_rgb, (target_l[idx][0], target_l[idx][1]), \
                                          (target_l[0][0], target_l[0][1]), (0, 0, 255), 2)
            else:
                target_img_rgb = cv2.line(target_img_rgb, (target_l[idx][0], target_l[idx][1]), \
                                          (target_l[idx+1][0], target_l[idx+1][1]), (0, 0, 255), 2)
        # template convex hull area
        temp_convexhull_area = calculatePolygonArea(template_l)
        targ_convexhull_area = calculatePolygonArea(target_l)

        temp_valid_area = calculateValidPixelsArea(template_img_gray)
        targ_valid_area = calculateValidPixelsArea(target_img_gray)

        # area ratio
        temp_convexhull_area_ratio = temp_convexhull_area / (template_img_gray.shape[0] * template_img_gray.shape[1]) \
                                     * 100.
        targ_convexhull_area_ratio = targ_convexhull_area / (target_img_gray.shape[0] * target_img_gray.shape[1]) \
                                     * 100.

        temp_valid_area_ratio = temp_valid_area / temp_convexhull_area * 100.
        targ_valid_area_ratio = targ_valid_area / targ_convexhull_area * 100.

        self.temp_convex_area_ratio_label.setText("%3.2f" % temp_convexhull_area_ratio)
        self.targ_convex_area_ratio_label_2.setText("%3.2f" % targ_convexhull_area_ratio)

        self.temp_valid_area_ratio_label.setText("%3.2f" % temp_valid_area_ratio)
        self.targ_valid_area_ratio_label_2.setText("%3.2f" % targ_valid_area_ratio)

        # display RGB images
        template_img_rgb = np.array(template_img_rgb)
        temp_qimg = rgb2qimage(template_img_rgb)
        temp_pix = QPixmap.fromImage(temp_qimg)

        self.template_ch_scene.addPixmap(temp_pix)
        self.template_ch_scene.setSceneRect(QRectF())
        self.template_ch_gview.fitInView(self.template_ch_scene.sceneRect(), Qt.KeepAspectRatio)
        self.template_ch_scene.update()
        self.whole_temp_ch_pix = temp_pix.copy()

        target_img_rgb = np.array(target_img_rgb)
        targ_qimg = rgb2qimage(target_img_rgb)
        targ_pix = QPixmap.fromImage(targ_qimg)

        self.target_ch_scene.addPixmap(targ_pix)
        self.target_ch_scene.setSceneRect(QRectF())
        self.target_ch_gview.fitInView(self.target_ch_scene.sceneRect(), Qt.KeepAspectRatio)
        self.target_ch_scene.update()
        self.whole_targ_ch_pix = targ_pix.copy()

        self.statusbar.showMessage("Convex hull successed!")

        del template_img_gray, template_img_rgb, temp_pix, temp_qimg, template_l, target_img_gray, target_img_rgb, \
            targ_pix, targ_qimg, target_l

    def projectBtn(self):
        print("Project button clicked!")

        if self.template_image_gray is None or self.target_image_gray is None:
            QMessageBox.information(self, "Grayscale image", "Grayscale image is None!")
            return
        # project on X-axis and Y-axis
        # grayscale images
        template_img_gray = self.template_image_gray.copy()
        target_img_gray = self.target_image_gray.copy()

        template_img_gray = np.array(template_img_gray, dtype=np.uint8)
        target_img_gray = np.array(target_img_gray, dtype=np.uint8)
        print(template_img_gray.shape)

        # X-axis and Y-axis statistics histogram
        temp_x_hist = np.zeros(template_img_gray.shape[1]); temp_y_hist = np.zeros(template_img_gray.shape[0])
        targ_x_hist = np.zeros(target_img_gray.shape[1]); targ_y_hist = np.zeros(target_img_gray.shape[0])

        # template X-axis and Y-axis histograms
        for y in range(template_img_gray.shape[0]):
            for x in range(template_img_gray.shape[1]):
                if template_img_gray[y][x] == 0:
                    temp_y_hist[y] += 1
                    temp_x_hist[x] += 1

        # target X-axis and Y-axis histograms
        for y in range(target_img_gray.shape[0]):
            for x in range(target_img_gray.shape[1]):
                if target_img_gray[y][x] == 0:
                    targ_y_hist[y] += 1
                    targ_x_hist[x] += 1

        # draw the plot of histograms
        self.temp_x_canvas.update_figure(temp_x_hist)
        self.temp_y_canvas.update_figure(temp_y_hist)
        self.targ_x_canvas.update_figure(targ_x_hist)
        self.targ_y_canvas.update_figure(targ_y_hist)

        # calcluate the mean and variance

        temp_x_mean = np.mean(temp_x_hist); temp_y_mean = np.mean(temp_y_hist)
        targ_x_mean = np.mean(targ_x_hist); targ_y_mean = np.mean(targ_y_hist)

        x_means = np.array([temp_x_hist, targ_x_hist])
        y_means = np.array([temp_y_hist, targ_y_hist])

        x_variance = np.var(x_means)
        y_variance = np.var(y_means)

        self.temp_x_mean_label.setText("%.2f" % temp_x_mean)
        self.temp_y_mean_label.setText("%.2f" % temp_y_mean)
        self.targ_x_mean_label.setText("%.2f" % targ_x_mean)
        self.targ_y_mean_label.setText("%.2f" % targ_y_mean)

        self.variance_x_label.setText("%.2f" % x_variance)
        self.variance_y_label.setText("%.2f" % y_variance)

        self.statusbar.showMessage("Project successed!")

        del template_img_gray, target_img_gray, temp_x_hist, temp_y_hist, targ_x_hist, targ_y_hist

    def storkeLayoutBtn(self):
        """
        Stroke layout function.
        :return:
        """
        print("Stroke layout button clicked!")

        #clear
        self.template_strokes_layout_scene.clear()
        self.target_strokes_layout_scene.clear()

        self.radical_temp_layout_pix = None
        self.radical_targ_layout_pix = None

        temp_strokes_rect = []
        targ_strokes_rect = []
        # templates stroke rectangle
        for st in self.template_image_strokes_name:
            path = self.template_image_stroke_path + st

            stroke_gray = cv2.imread(path, 0)
            stroke_rect = getSingleMaxBoundingBoxOfImage(stroke_gray)
            if stroke_rect is None:
                continue
            temp_strokes_rect.append(stroke_rect)
            del stroke_gray
        # target stoke rectangle
        for st in self.target_image_strokes_name:
            path = self.target_image_stroke_path + st

            stroke_gray = cv2.imread(path, 0)
            stroke_rect = getSingleMaxBoundingBoxOfImage(stroke_gray)
            if stroke_rect is None:
                continue
            targ_strokes_rect.append(stroke_rect)
            del stroke_gray

        # display all stroke rectangles on RGB images
        temp_img_rgb = cv2.cvtColor(self.template_image_gray_no_resize, cv2.COLOR_GRAY2RGB)
        targ_img_rgb = cv2.cvtColor(self.target_image_gray_no_resize, cv2.COLOR_GRAY2RGB)

        for idx in range(len(temp_strokes_rect)):
            rect_color = None
            rt = temp_strokes_rect[idx]
            if idx % 3 == 0:
                rect_color = (0, 0, 255)
            elif idx % 3 == 1:
                rect_color = (0, 255, 0)
            elif idx % 3 == 2:
                rect_color = (255, 0, 0)
            cv2.rectangle(temp_img_rgb, (rt[0], rt[1]), (rt[0] + rt[2], rt[1] + rt[3]), rect_color, 1)

        for idx in range(len(targ_strokes_rect)):
            rect_color = None
            rt = targ_strokes_rect[idx]
            if idx % 3 == 0:
                rect_color = (0, 0, 255)
            elif idx % 3 == 1:
                rect_color = (0, 255, 0)
            elif idx % 3 == 2:
                rect_color = (255, 0, 0)
            cv2.rectangle(targ_img_rgb, (rt[0], rt[1]), (rt[0] + rt[2], rt[1] + rt[3]), rect_color, 1)
        # show image on gview
        temp_img_rgb = np.array(temp_img_rgb)
        targ_img_rgb = np.array(targ_img_rgb)

        temp_qimg = rgb2qimage(temp_img_rgb)
        targ_qimg = rgb2qimage(targ_img_rgb)

        temp_pix = QPixmap.fromImage(temp_qimg)
        targ_pix = QPixmap.fromImage(targ_qimg)

        self.template_strokes_layout_scene.addPixmap(temp_pix)
        self.template_strokes_layout_scene.setSceneRect(QRectF())
        self.template_strokes_layout_gview.fitInView(self.template_strokes_layout_scene.sceneRect(), Qt.KeepAspectRatio)
        self.template_strokes_layout_scene.update()
        self.radical_temp_layout_pix = temp_pix.copy()

        self.target_strokes_layout_scene.addPixmap(targ_pix)
        self.target_strokes_layout_scene.setSceneRect(QRectF())
        self.target_strokes_layout_gview.fitInView(self.target_strokes_layout_scene.sceneRect(), Qt.KeepAspectRatio)
        self.target_strokes_layout_scene.update()
        self.radical_targ_layout_pix = targ_pix.copy()

        self.statusbar.showMessage("Stroke layout successed!")

        del temp_strokes_rect, temp_img_rgb, temp_qimg, temp_pix, targ_strokes_rect, targ_img_rgb, targ_qimg, targ_pix

    def strokes_item_selected(self, qModelIndex):
        """
        Strokes item selected.
        :return:
        """
        self.strokes_cr_scene.clear()
        self.stroke_cr_pix = None

        print("Stroke %d selected!" % qModelIndex.row())
        stroke_name = self.template_image_strokes_name[qModelIndex.row()]

        self.stroke_selected_name = stroke_name

        temp_stroke_path = self.template_image_stroke_path + stroke_name
        targ_stroke_path = self.target_image_stroke_path + stroke_name

        temp_img_gray = cv2.imread(temp_stroke_path, 0)
        targ_img_gray = cv2.imread(targ_stroke_path, 0)

        _, temp_img_gray = cv2.threshold(temp_img_gray, 127, 255, cv2.THRESH_BINARY)
        _, targ_img_gray = cv2.threshold(targ_img_gray, 127, 255, cv2.THRESH_BINARY)

        # resize
        temp_img_gray, targ_img_gray = resizeImages(temp_img_gray, targ_img_gray)

        # shit the target image to get the max cr
        targ_img_gray = shiftImageWithMaxCR(temp_img_gray, targ_img_gray)

        # set select stroke after resizing and shifting
        self.template_stroke_selected_gray = temp_img_gray.copy()
        self.target_stroke_selected_gray = targ_img_gray.copy()

        cr = calculateCoverageRate(temp_img_gray, targ_img_gray)

        self.strokes_cr_label.setText("%.2f" % cr)

        # cover two image in one RGB image
        cr_rgb = coverTwoImages(temp_img_gray, targ_img_gray)

        # display RGB image
        cr_rgb_qimg = rgb2qimage(cr_rgb)
        cr_rgb_pix = QPixmap.fromImage(cr_rgb_qimg)
        self.strokes_cr_scene.addPixmap(cr_rgb_pix)
        self.strokes_cr_scene.setSceneRect(QRectF())
        self.strokes_cr_gview.fitInView(self.strokes_cr_scene.sceneRect(), Qt.KeepAspectRatio)
        self.strokes_cr_scene.update()
        self.stroke_cr_pix = cr_rgb_pix.copy()

        self.statusbar.showMessage("Stroke cr successed!")

        del temp_img_gray, targ_img_gray, cr_rgb, cr_rgb_pix, cr_rgb_qimg

    def exitBtn(self):
        print("Exit button clicked!")
        qApp = QApplication.instance()
        sys.exit(qApp.exec_())

    def wholeMainTabWidgetOnChange(self, i):
        print("whole tab index: %d" % i)

        if i == 0:
            # whole cr tab click
            if self.whole_cr_pix:
                self.cover_scene.addPixmap(self.whole_cr_pix)
                self.cover_scene.setSceneRect(QRectF())
                self.cover_gview.fitInView(self.cover_scene.sceneRect(), Qt.KeepAspectRatio)
                self.cover_scene.update()
        elif i == 1:
            # whole goc tab click
            if self.whole_temp_goc_pix and self.whole_targ_goc_pix and self.whole_combine_goc_pix:
                self.template_cog_scene.addPixmap(self.whole_temp_goc_pix)
                self.template_cog_scene.setSceneRect(QRectF())
                self.cog_template_gview.fitInView(self.template_cog_scene.sceneRect(), Qt.KeepAspectRatio)
                self.template_cog_scene.update()

                self.target_cog_scene.addPixmap(self.whole_targ_goc_pix)
                self.target_cog_scene.setSceneRect(QRectF())
                self.cog_target_gview.fitInView(self.target_cog_scene.sceneRect(), Qt.KeepAspectRatio)
                self.target_cog_scene.update()

                self.cog_comparison_scene.addPixmap(self.whole_combine_goc_pix)
                self.cog_comparison_scene.setSceneRect(QRectF())
                self.cog_comparison_gview.fitInView(self.cog_comparison_scene.sceneRect(), Qt.KeepAspectRatio)
                self.cog_comparison_scene.update()
        elif i == 2:
            # whole ch tab click
            if self.whole_temp_ch_pix and self.whole_targ_ch_pix:
                self.template_ch_scene.addPixmap(self.whole_temp_ch_pix)
                self.template_ch_scene.setSceneRect(QRectF())
                self.template_ch_gview.fitInView(self.template_ch_scene.sceneRect(), Qt.KeepAspectRatio)
                self.template_ch_scene.update()

                self.target_ch_scene.addPixmap(self.whole_targ_ch_pix)
                self.target_ch_scene.setSceneRect(QRectF())
                self.target_ch_gview.fitInView(self.target_ch_scene.sceneRect(), Qt.KeepAspectRatio)
                self.target_ch_scene.update()

    def radicalMainTabWidgetOnChange(self, i):
        print("radical tab index: %d" % i)
        if i == 1:
            # radical layout comparison
            if self.radical_temp_layout_pix and self.radical_targ_layout_pix:
                self.template_strokes_layout_scene.addPixmap(self.radical_temp_layout_pix)
                self.template_strokes_layout_scene.setSceneRect(QRectF())
                self.template_strokes_layout_gview.fitInView(self.template_strokes_layout_scene.sceneRect(),
                                                             Qt.KeepAspectRatio)
                self.template_strokes_layout_scene.update()

                self.target_strokes_layout_scene.addPixmap(self.radical_targ_layout_pix)
                self.target_strokes_layout_scene.setSceneRect(QRectF())
                self.target_strokes_layout_gview.fitInView(self.target_strokes_layout_scene.sceneRect(),
                                                           Qt.KeepAspectRatio)
                self.target_strokes_layout_scene.update()

    def strokeMainTabWidgetOnChange(self, i):
        print("stroke tab index: %d" % i)
        if i == 0:
            # stroke cr
            if self.stroke_cr_pix:
                self.strokes_cr_scene.addPixmap(self.stroke_cr_pix)
                self.strokes_cr_scene.setSceneRect(QRectF())
                self.strokes_cr_gview.fitInView(self.strokes_cr_scene.sceneRect(), Qt.KeepAspectRatio)
                self.strokes_cr_scene.update()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def update_figure(self, data):
        self.axes.plot(data)
        self.draw()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = CalligraphyComparisonGUI()
    mainwindow.show()
    sys.exit(app.exec_())