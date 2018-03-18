# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'strokeExtractingMainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import math
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1278, 667)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 171, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.open_btn = QPushButton(self.verticalLayoutWidget)
        self.open_btn.setObjectName("open_btn")
        self.open_btn.clicked.connect(self.openBtn)
        self.verticalLayout.addWidget(self.open_btn)

        self.extract_btn = QPushButton(self.verticalLayoutWidget)
        self.extract_btn.setObjectName("extract_btn")
        self.extract_btn.clicked.connect(self.extractBtn)
        self.verticalLayout.addWidget(self.extract_btn)

        self.clear_btn = QPushButton(self.verticalLayoutWidget)
        self.clear_btn.setObjectName("clear_btn")
        self.clear_btn.clicked.connect(self.clearBtn)
        self.verticalLayout.addWidget(self.clear_btn)

        self.image_pix = QPixmap()
        self.temp_image_pix = QPixmap()

        self.scene = GraphicsScene()

        self.image_view = QGraphicsView(self.centralwidget)
        self.image_view.setGeometry(QRect(200, 10, 1061, 591))
        self.image_view.setObjectName("image_view")
        self.image_view.setScene(self.scene)
        self.image_view.setMouseTracking(True)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 1278, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

        self.lastPoint = QPoint()
        self.endPoint = QPoint()

        self.image_gray = None


    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.open_btn.setText(_translate("MainWindow", "Open"))
        self.extract_btn.setText(_translate("MainWindow", "Extracting"))
        self.clear_btn.setText(_translate("MainWindow", "Clear"))

    def openBtn(self):
        print("Open button clicked!")
        self.scene.clear()
        filename, _ = QFileDialog.getOpenFileName(None, "Open file", QDir.currentPath())
        if filename:
            image = QImage(filename)
            if image.isNull():
                QMessageBox.information(self, "Image viewer", "Cannot load %s." % filename)
                return
            print(filename)

            # grayscale image
            img_ = cv2.imread(filename, 0)
            _, img_ = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY)
            self.image_gray = img_.copy()

            self.image_pix = QPixmap.fromImage(image)
            self.temp_image_pix = self.image_pix.copy()
            self.scene.addPixmap(self.image_pix)
            self.scene.update()
            

    def extractBtn(self):
        print("Extract button clicked")
        if self.image_gray is None:
            QMessageBox.information(self, "Grayscale image is None!")

        if self.scene.points is None or len(self.scene.points) == 0:
            # No points selected, return all image
            cv2.imwrite("all.png", self.image_gray)
        else:
            stroke = extractStorkeByPolygon(self.image_gray, self.scene.points)
            cv2.imwrite("stroke.png", stroke)


        print("number of points: %d" % len(self.scene.points))



    def clearBtn(self):
        print("Clear !")

        # remove existing points
        self.scene.lastPoint = QPoint()
        self.scene.endPoint = QPoint()
        self.scene.points = []

        # remove points in image
        self.image_pix = self.temp_image_pix.copy()
        self.scene.addPixmap(self.image_pix)
        self.scene.update()



class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.lastPoint = QPoint()
        self.endPoint = QPoint()

        self.points = []

    def setOption(self, opt):
        self.opt = opt

    def mousePressEvent(self, event):
        print(event.scenePos())
        pen = QPen(Qt.red)
        brush = QBrush(Qt.red)
        x = event.scenePos().x()
        y = event.scenePos().y()

        self.addEllipse(x, y, 4, 4, pen, brush)
        self.endPoint = event.scenePos()
        self.points.append((x, y))

    def mouseReleaseEvent(self, event):
        pen = QPen(Qt.red)

        if self.lastPoint.x() != 0.0 and self.lastPoint.y() != 0.0:
            self.addLine(self.endPoint.x(), self.endPoint.y(), self.lastPoint.x(), self.lastPoint.y(), pen)

        self.lastPoint = self.endPoint

def extractStorkeByPolygon(image, polygon):

    image_ = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if not ray_tracing_method(x, y, polygon):
                image_[y][x] = 255

    return image_

# Ray tracing check point in polygon
def ray_tracing_method(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside