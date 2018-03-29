import sys
import math
import cv2
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from smoothmanuallymainwindow import Ui_MainWindow
from utils.Functions import getContourOfImage


class SmoothManuallyGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(SmoothManuallyGUI, self).__init__()
        self.setupUi(self)

        # init GUI
        self.scene = GraphicsScene()
        self.contour_gview.setScene(self.scene)

        self.image_pix = QPixmap()
        self.temp_image_pix = QPixmap()

        self.contour_pix = QPixmap()
        self.temp_contour_pix = QPixmap()


        # data
        self.image_path = ""
        self.image_name = ""

        self.feature_points = []

        self.image_gray = None
        self.contour_gray = None


        # add listener
        self.open_btn.clicked.connect(self.openBtn)
        self.clear_btn.clicked.connect(self.clearBtn)
        self.contour_btn.clicked.connect(self.contourBtn)
        self.smooth_btn.clicked.connect(self.smoothBtn)
        self.autosmooth_btn.clicked.connect(self.autoSmoothBtn)
        self.save_btn.clicked.connect(self.saveBtn)
        self.exit_btn.clicked.connect(self.exitBtn)

    def openBtn(self):
        """
        Open button
        :return:
        """
        print("Open button clicked!")
        self.scene.clear()

        filename, _ = QFileDialog.getOpenFileName(None, "Open file", QDir.currentPath())
        if filename:
            # image file path and name
            self.image_path = filename
            self.image_name = os.path.splitext(os.path.basename(filename))[0]

            qimage = QImage(filename)
            if qimage.isNull():
                QMessageBox.information(self, "Image viewer", "Cannot not load %s." % filename)
                return
            # grayscale image
            img_ = cv2.imread(filename, 0)
            _, img_ = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY)
            self.image_gray = img_.copy()

            self.image_pix = QPixmap.fromImage(qimage)
            self.temp_image_pix = self.image_pix.copy()
            self.scene.addPixmap(self.image_pix)
            self.scene.update()
            self.statusbar.showMessage("Open image %s successed!" % self.image_name)

            # clean
            del img_

    def clearBtn(self):
        """
        Clear button clicked
        :return:
        """
        print("Clear button clicked")
        self.scene.addPixmap(self.image_pix)
        self.scene.update()
        self.statusbar.showMessage("Clear successed!")

    def contourBtn(self):
        """
        Obtain the contour of image
        :return:
        """
        print("Contour button clicked")
        contour_ = getContourOfImage(self.image_gray)

        self.contour_gray = contour_.copy()
        qimg = QImage(contour_.data, contour_.shape[1], contour_.shape[0], contour_.shape[1], QImage.Format_Indexed8)

        self.contour_pix = QPixmap.fromImage(qimg)
        self.temp_contour_pix = self.contour_pix.copy()
        self.scene.addPixmap(self.contour_pix)
        self.scene.update()

        del contour_


    def smoothBtn(self):
        """
        Smooth button clicked!
        :return:
        """
        print("Smooth button clicked")

    def autoSmoothBtn(self):
        """
        Auto-smooth button clicked
        :return:
        """
        print("Auto-smooth button clicked")

    def saveBtn(self):
        """
        Save button
        :return:
        """
        print("Save button clicked")

    def exitBtn(self):
        """
        Exit
        :return:
        """
        qApp = QApplication.instance()
        sys.exit(qApp.exec_())


class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.points = []

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

        # add point
        self.addEllipse(x, y, 3, 3, pen, brush)

        self.points.append((x, y))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = SmoothManuallyGUI()
    mainWindow.show()
    sys.exit(app.exec_())