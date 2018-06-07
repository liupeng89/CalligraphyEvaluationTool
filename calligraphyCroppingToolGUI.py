import sys
import math
import cv2
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from calligraphyCroppingTool.charactersegmentationmainwindow import Ui_MainWindow
from utils.Functions import getAllMiniBoundingBoxesOfImage, getCenterOfRectangles, combineRectangles, rgb2qimage
from calligraphyCroppingTool.tools import filterBoxWithWidth, removeContainedBoxes


class CharacterSegmentationMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(CharacterSegmentationMainWindow, self).__init__()
        self.setupUi(self)

        # image data
        self.image_rgb = None
        self.image_gray = None
        self.image_binary = None
        self.image_characters = []
        self.characters_name = []

        # scene
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(Qt.gray)
        self.image_gview.setScene(self.scene)

        self.char_slm = QStringListModel()
        self.char_slm.setStringList(self.characters_name)
        self.characters_list.setModel(self.char_slm)
        self.characters_list.clicked.connect(self.charsListView_clicked)

        self.image_pix = QImage()
        self.temp_image_pix = QImage()

        self.image_name = ""
        self.image_path = ""

        # add listener
        self.open_btn.clicked.connect(self.openBtn)
        self.grayscale_btn.clicked.connect(self.grayscaleBtn)
        self.convert_btn.clicked.connect(self.convertBtn)

        self.segmentation_btn.clicked.connect(self.segmentationBtn)
        self.extract_btn.clicked.connect(self.extractBtn)
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
            self.scene.setSceneRect(QRectF())
            self.image_gview.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.scene.update()

            self.statusbar.showMessage("Open image: %s successed!" % self.image_name)

            del img_rgb, qimage

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
        self.scene.setSceneRect(QRectF())
        self.image_gview.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.scene.update()

        self.statusbar.showMessage("Grayscale processing successed!")

        del img_rgb, img_gray, qimg

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
        self.scene.setSceneRect(QRectF())
        self.image_gview.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.scene.update()

        self.statusbar.showMessage("Conveting color processing successed!")

        del img_gray, qimg

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
        self.scene.setSceneRect(QRectF())
        self.image_gview.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.scene.update()

        self.threshold_label.setText(str(binary_threshold))

        del img_gray, img_binary, qimg

    def segmentationBtn(self):
        """
        Characters segmentation.
        :return:
        """
        print("Segmentation button clicked!")

        self.scene.clear()
        self.image_characters = []
        self.characters_name = []

        img_rgb = self.image_rgb.copy()

        # distance threshold
        dist_threshold = int(self.thre_dist_ledit.text())
        width_threshold = int(self.thre_width_ledit.text())

        img_binary = self.image_binary.copy()

        boxes = getAllMiniBoundingBoxesOfImage(img_binary)
        # filter boxes with threshold of width
        boxes = filterBoxWithWidth(boxes, width_threshold)
        print("original boxes len: %d" % len(boxes))

        # remove inside rectangles
        boxes_noinside = removeContainedBoxes(boxes)

        # cluster rectangles based on the distance threshold
        lines = []
        for i in range(len(boxes_noinside)):
            rect_item = [i]

            ct_rect_i = getCenterOfRectangles(boxes_noinside[i])
            start_index = i
            end_index = start_index

            for j in range(len(boxes_noinside)):
                if i == j:
                    continue
                ct_rect_j = getCenterOfRectangles(boxes_noinside[j])

                dist = math.sqrt((ct_rect_j[0] - ct_rect_i[0]) ** 2 + (ct_rect_j[1] - ct_rect_i[1]) ** 2)
                if dist <= dist_threshold:
                    rect_item.append(j)

                    end_index = j
                    lines.append([start_index, end_index])
            if end_index == start_index:
                lines.append([start_index, end_index])

        # cluster based on the lines
        rects = []
        for i in range(len(lines)):
            line = lines[i]
            if line[0] == line[1]:
                rects.append([line[0]])
            else:
                new_set = set(line)
                for j in range(len(lines)):
                    if i == j:
                        continue
                    set_j = set(lines[j])
                    if len(new_set.intersection(set_j)) != 0:
                        new_set = new_set.union(set_j)
                if list(new_set) not in rects:
                    rects.append(list(new_set))

        # remove the repeat items
        rects_ = []
        repet_id = []
        for i in range(len(rects)):
            if i in repet_id:
                continue
            repet_id.append(i)
            len1 = len(repet_id)
            for j in range(len(rects)):
                if i == j:
                    continue
                if len(set(rects[j]).intersection(set(rects[i]))) != 0:
                    new_set = list(set(rects[j]).union(set(rects[i])))
                    rects_.append(new_set)
                    repet_id.append(j)
            if len1 == len(repet_id):
                rects_.append(rects[i])

        del rects
        rects = rects_.copy()
        del rects_

        for i in range(len(rects)):
            new_rect = combineRectangles(boxes_noinside, rects[i])
            # add border of rectangle with 5 pixels
            new_r_x = 0 if new_rect[0]-5 < 0 else new_rect[0]-5
            new_r_y = 0 if new_rect[1]-5 < 0 else new_rect[1]-5
            new_r_w = img_binary.shape[1]-new_rect[0] if new_rect[0]+new_rect[2]+10 > img_binary.shape[1] else new_rect[2]+10
            new_r_h = img_binary.shape[0]-new_rect[0] if new_rect[1]+new_rect[3]+10 > img_binary.shape[0] else new_rect[3]+10

            cv2.rectangle(img_rgb, (new_r_x, new_r_y), (new_r_x + new_r_w, new_r_y + new_r_h),
                          (255, 0, 0), 1)
            self.image_characters.append((new_r_x, new_r_y, new_r_w, new_r_h))
            self.characters_name.append("character_" + str(i+1))

        # display RGB image
        img_rgb = np.array(img_rgb)
        qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
        self.image_pix = QPixmap.fromImage(qimg)
        self.temp_image_pix = self.image_pix.copy()

        self.scene.addPixmap(self.image_pix)
        self.scene.setSceneRect(QRectF())
        self.image_gview.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.scene.update()

        self.char_slm.setStringList(self.characters_name)

        self.statusbar.showMessage("Segmentation processing successed!")
        del qimg, img_binary, img_rgb, boxes_noinside, boxes

    def charsListView_clicked(self, qModelIndex):
        """
        :param qModelIndex:
        :return:
        """
        print("list item %d clicked" % qModelIndex.row())
        self.scene.clear()
        if len(self.image_characters) == 0:
            return

        rect = self.image_characters[qModelIndex.row()]

        img_rect = self.image_rgb[rect[1]: rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        img_rect = np.array(img_rect, dtype=np.uint8)
        qimg = rgb2qimage(img_rect)

        self.image_pix = QPixmap.fromImage(qimg)
        self.temp_image_pix = self.image_pix.copy()

        self.scene.addPixmap(self.image_pix)
        self.scene.setSceneRect(QRectF())
        self.image_gview.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.scene.update()

        del rect, img_rect, qimg

    def extractBtn(self):
        """
        Extract button clicked.
        :return:
        """
        print("Extract button clicked!")
        if len(self.image_characters) == 0:
            return
        # save path
        fileName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(fileName)

        # extract all characters
        for i in range(len(self.image_characters)):
            name_str = "character_" + str(i+1)
            path = os.path.join(fileName, self.image_name + "_" + name_str + ".png")

            # rect
            rect = self.image_characters[i]

            img_rect = self.image_rgb[rect[1]: rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            img_rect = np.array(img_rect)

            cv2.imwrite(path, img_rect)
        self.statusbar.showMessage("Save characters successed!")

        del rect, img_rect

    def exitBtn(self):
        """
            Exiting button clicked function.
        :return:
        """
        qApp = QApplication.instance()
        sys.exit(qApp.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CharacterSegmentationMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())