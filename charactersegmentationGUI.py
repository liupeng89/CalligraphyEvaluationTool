import sys
import math
import cv2
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from charactersegmentationmainwindow import Ui_MainWindow
from utils.Functions import getAllMiniBoundingBoxesOfImage, getCenterOfRectangles, combineRectangles


class CharacterSegmentationMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(CharacterSegmentationMainWindow, self).__init__()
        self.setupUi(self)

        # image data
        self.image_rgb = None
        self.image_gray = None
        self.image_binary = None
        self.image_characters = []

        # scene
        self.scene = GraphicsScene()
        self.image_gview.setScene(self.scene)

        self.image_pix = QImage()
        self.temp_image_pix = QImage()

        self.image_name = ""
        self.image_path = ""

        # add listener
        self.open_btn.clicked.connect(self.openBtn)
        self.grayscale_btn.clicked.connect(self.grayscaleBtn)
        self.convert_btn.clicked.connect(self.convertBtn)

        self.segmentation_btn.clicked.connect(self.segmentationBtn)
        self.exit_btn.clicked.connect(self.exitBtn)
        self.binary_threshold_slider.valueChanged.connect(self.binary_threshold_valuechange)

    def openBtn(self):
        """
        Open button clicked!
        :return:
        """
        print("Open button clicked!")
        self.scene.clear()

        filename, _ = QFileDialog.getOpenFileName(None, "Open file", QDir.currentPath())
        if filename:
            # image file path
            self.image_path = filename
            self.image_name = os.path.splitext(os.path.basename(filename))[0]

            qimage = QImage(filename)
            if qimage.isNull():
                QMessageBox.information(self, "Image viewer", "Cannot load %s." % filename)
                return
            # RGB image
            img_rgb = cv2.imread(filename)
            self.image_rgb = img_rgb.copy()

            self.image_pix = QPixmap.fromImage(qimage)
            self.temp_image_pix = self.image_pix.copy()

            self.scene.addPixmap(self.image_pix)
            self.scene.update()

            self.statusbar.showMessage("Open image: %s successed!" % self.image_name)

            del img_rgb
            del qimage

    def grayscaleBtn(self):
        """
        Converting RGB image to grayscale image.
        :return:
        """
        print("Grayscale button clicked!")
        img_rgb = self.image_rgb.copy()

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        self.image_gray = img_gray.copy()

        # display grayscale image
        qimg = QImage(img_gray.data, img_gray.shape[1], img_gray.shape[0], img_gray.shape[1], QImage.Format_Indexed8)

        self.image_pix = QPixmap.fromImage(qimg)
        self.temp_image_pix = self.image_pix.copy()

        self.scene.addPixmap(self.image_pix)
        self.scene.update()

        self.statusbar.showMessage("Grayscale processing successed!")

        del img_rgb
        del img_gray
        del qimg

    def convertBtn(self):
        """
        Convert color processing.
        :return:
        """
        print("Convert button clicked!")
        img_gray = self.image_gray.copy()

        img_gray = 255 - img_gray
        img_gray = np.array(img_gray, dtype=np.uint8)
        self.image_gray = img_gray.copy()

        # display grayscale image
        qimg = QImage(img_gray.data, img_gray.shape[1], img_gray.shape[0], img_gray.shape[1], QImage.Format_Indexed8)

        self.image_pix = QPixmap.fromImage(qimg)
        self.temp_image_pix = self.image_pix.copy()

        self.scene.addPixmap(self.image_pix)
        self.scene.update()

        self.statusbar.showMessage("Conveting color processing successed!")

        del img_gray
        del qimg

    def binary_threshold_valuechange(self):
        """
        Binary with threshold.
        :return:
        """
        binary_threshold = self.binary_threshold_slider.value()

        if self.image_gray is None:
            return

        img_gray = self.image_gray.copy()
        _, img_binary = cv2.threshold(img_gray, binary_threshold, 255, cv2.THRESH_BINARY)

        self.image_binary = img_binary.copy()

        # display binary image
        qimg = QImage(img_binary.data, img_binary.shape[1], img_binary.shape[0], img_binary.shape[1],
                      QImage.Format_Indexed8)

        self.image_pix = QPixmap.fromImage(qimg)
        self.temp_image_pix = self.image_pix.copy()

        self.scene.addPixmap(self.image_pix)
        self.scene.update()

        self.threshold_label.setText(str(binary_threshold))

        del img_gray
        del img_binary
        del qimg

    def segmentationBtn(self):
        """
        Characters segmentation.
        :return:
        """
        print("Segmentation button clicked!")

        # distance threshold
        dist_threshold = int(self.thre_dist_ledit.text())

        img_binary = self.image_binary.copy()

        boxes = getAllMiniBoundingBoxesOfImage(img_binary)
        boxes_ = []
        for box in boxes:
            if box[2] < dist_threshold or box[3] < dist_threshold:
                continue
            boxes_.append(box)

        del boxes
        boxes = boxes_.copy()
        del boxes_
        print("original boxes len: %d" % len(boxes))

        # remove inside rectangles
        inside_id = []
        for i in range(len(boxes)):
            ri_x = boxes[i][0]
            ri_y = boxes[i][1]
            ri_w = boxes[i][2]
            ri_h = boxes[i][3]

            for j in range(len(boxes)):
                if i == j or j in inside_id:
                    continue
                rj_x = boxes[j][0]
                rj_y = boxes[j][1]
                rj_w = boxes[j][2]
                rj_h = boxes[j][3]

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

        # cluster rectangles based on the distance threshold
        rect_clustor = []
        for i in range(len(boxes_noinside)):
            rect_item = []
            rect_item.append(i)

            ct_rect_i = getCenterOfRectangles(boxes_noinside[i])

            for j in range(len(boxes_noinside)):
                ct_rect_j = getCenterOfRectangles(boxes_noinside[j])

                dist = math.sqrt((ct_rect_j[0] - ct_rect_i[0]) * (ct_rect_j[0] - ct_rect_i[0]) + (ct_rect_j[1] - ct_rect_i[1]) * (
                            ct_rect_j[1] - ct_rect_i[1]))
                if dist <= dist_threshold and j not in rect_item:
                    rect_item.append(j)
            rect_clustor.append(rect_item)
        print(rect_clustor)

        # merge the cluster
        final_cluster = []
        used_index = []
        for i in range(len(rect_clustor)):
            if i in used_index:
                continue
            new_cluster = rect_clustor[i]

            # merge
            for j in range(i+1, len(rect_clustor)):
                if len(set(new_cluster).intersection(set(rect_clustor[j]))) == 0:
                    continue
                new_cluster = list(set(new_cluster).union(set(rect_clustor[j])))
                used_index.append(j)
            final_cluster.append(new_cluster)
        print(final_cluster)

        img_rgb = self.image_rgb.copy()
        for i in range(len(final_cluster)):
            new_rect = combineRectangles(boxes_noinside, final_cluster[i])
            cv2.rectangle(img_rgb, (new_rect[0], new_rect[1]), (new_rect[0] + new_rect[2], new_rect[1] + new_rect[3]),
                          (0, 0, 255), 1)

        # display RGB image
        qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[1], QImage.Format_RGB32)
        self.image_pix = QPixmap.fromImage(qimg)
        self.temp_image_pix = self.image_pix.copy()

        self.scene.addPixmap(self.image_pix)
        self.scene.update()

        self.statusbar.showMessage("Segmentation processing successed!")
        del qimg
        del img_rgb
        del img_binary
        del boxes_noinside
        del boxes
        del final_cluster
        del rect_clustor
        del used_index
        del inside_id

    def exitBtn(self):
        """
            Exiting button clicked function.
        :return:
        """
        qApp = QApplication.instance()
        sys.exit(qApp.exec_())


class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.lastPoint = QPoint()
        self.endPoint = QPoint()

        self.points = []
        self.strokes = []
        self.T_DISTANCE = 10

    def setOption(self, opt):
        self.opt = opt

    def mousePressEvent(self, event):
        """
            Mouse press clicked!
        :param event:
        :return:
        """
        pen = QPen(Qt.red)
        brush = QBrush(Qt.red)
        x = event.scenePos().x()
        y = event.scenePos().y()

        if len(self.points) == 0:
            self.addEllipse(x, y, 2, 2, pen, brush)
            self.endPoint = event.scenePos()
        else:
            x0 = self.points[0][0]
            y0 = self.points[0][1]

            dist = math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
            if dist < self.T_DISTANCE:
                pen_ = QPen(Qt.green)
                brush_ = QBrush(Qt.green)
                self.addEllipse(x0, y0, 2, 2, pen_, brush_)
                self.endPoint = event.scenePos()
                x = x0; y = y0
            else:
                self.addEllipse(x, y, 4, 4, pen, brush)
                self.endPoint = event.scenePos()
        self.points.append((x, y))

    def mouseReleaseEvent(self, event):
        """
            Mouse release event!
        :param event:
        :return:
        """
        pen = QPen(Qt.red)

        if self.lastPoint.x() != 0.0 and self.lastPoint.y() != 0.0:
            self.addLine(self.endPoint.x(), self.endPoint.y(), self.lastPoint.x(), self.lastPoint.y(), pen)

        self.lastPoint = self.endPoint


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CharacterSegmentationMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())