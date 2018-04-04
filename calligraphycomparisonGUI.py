import sys
import math
import cv2
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from calligraphycomparisonmainwindow import Ui_MainWindow


class CalligraphyComparisonGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(CalligraphyComparisonGUI, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = CalligraphyComparisonGUI()
    mainwindow.show()
    sys.exit(app.exec_())