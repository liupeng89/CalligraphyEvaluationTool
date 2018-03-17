# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'strokeExtractingMainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1278, 667)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 171, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.open_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.open_btn.setObjectName("open_btn")
        self.verticalLayout.addWidget(self.open_btn)
        self.extract_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.extract_btn.setObjectName("extract_btn")
        self.verticalLayout.addWidget(self.extract_btn)
        self.exit_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.exit_btn.setObjectName("exit_btn")
        self.verticalLayout.addWidget(self.exit_btn)
        self.image_view = QtWidgets.QGraphicsView(self.centralwidget)
        self.image_view.setGeometry(QtCore.QRect(200, 10, 1061, 591))
        self.image_view.setObjectName("image_view")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1278, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.open_btn.setText(_translate("MainWindow", "Open"))
        self.extract_btn.setText(_translate("MainWindow", "Extracting"))
        self.exit_btn.setText(_translate("MainWindow", "Exit"))

