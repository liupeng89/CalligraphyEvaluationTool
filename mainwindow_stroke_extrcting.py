import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint


class Winform(QWidget):
    def __init__(self, parent=None):
        super(Winform, self).__init__(parent)
        self.setWindowTitle("绘图例子")


        self.pix = QPixmap()

        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.points = []

        self.initUi()

    def initUi(self):
        image = QImage("ben.png")
        # 窗口大小设置为600*500
        self.resize(600, 500)
        # 画布大小为400*400，背景为白色
        # self.pix = QPixmap(400, 400)
        # self.pix.fill(Qt.white)
        self.pix = QPixmap.fromImage(image)

    def paintEvent(self, event):
        pp = QPainter(self.pix)

        # Draw circle in end point
        pp.drawEllipse(self.endPoint, 2, 2)

        # 根据鼠标指针前后两个位置绘制直线
        pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

        print("len: %d" % len(self.points))

    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            if self.lastPoint == None and self.endPoint == None:
                self.lastPoint = event.pos()
                self.endPoint = event.pos()
            x = event.pos().x()
            y = event.pos().y()

            self.points.append((x, y))
            # self.update()
            # self.lastPoint = event.pos()
            # self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        pass
        # if event.buttons() and Qt.LeftButton:
        #     self.endPoint = event.pos()
        #     # 进行重新绘制
        #     self.update()

    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()
            # self.lastPoint = self.endPoint




if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Winform()
    form.show()
    sys.exit(app.exec_())