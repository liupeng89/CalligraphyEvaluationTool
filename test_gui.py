# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
# from PyQt5.QtGui import QPainter, QPixmap, QImage
# from PyQt5.QtCore import Qt, QPoint
#
#
# class Winform(QWidget):
#     def __init__(self, parent=None):
#         super(Winform, self).__init__(parent)
#         layout = QVBoxLayout()
#
#         self.setWindowTitle("双缓冲绘图例子")
#         self.pix = QPixmap()
#         self.lastPoint = QPoint()
#         self.endPoint = QPoint()
#
#         self.points = []
#
#         # 辅助画布
#         self.tempPix = QPixmap()
#         # 标志是否正在绘图
#         self.isDrawing = False
#         self.initUi()
#
#     def initUi(self):
#         # 窗口大小设置为600*500
#         self.resize(600, 500);
#         # 画布大小为400*400，背景为白色
#         self.pix = QPixmap(400, 400);
#         image = QImage("ben.png")
#         self.pix = QPixmap.fromImage(image)
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         x = self.lastPoint.x()
#         y = self.lastPoint.y()
#
#         x1 = self.endPoint.x()
#         y1 = self.endPoint.y()
#
#         w = self.endPoint.x() - x
#         h = self.endPoint.y() - y
#
#         # 如果正在绘图，就在辅助画布上绘制
#         if self.isDrawing:
#             # 将以前pix中的内容复制到tempPix中，保证以前的内容不消失
#             self.tempPix = self.pix
#             pp = QPainter(self.tempPix)
#             # pp.drawRect(x, y, w, h)
#             pp.drawLine(self.lastPoint, self.endPoint)
#             painter.drawPixmap(0, 0, self.tempPix)
#         else:
#             pp = QPainter(self.pix)
#             # pp.drawRect(x, y, w, h)
#             pp.drawLine(self.lastPoint, self.endPoint)
#             painter.drawPixmap(0, 0, self.pix)
#
#     def mousePressEvent(self, event):
#         # 鼠标左键按下
#         if event.button() == Qt.LeftButton:
#             self.lastPoint = event.pos()
#             self.endPoint = self.lastPoint
#             self.isDrawing = True
#
#     def mouseReleaseEvent(self, event):
#         # 鼠标左键释放
#         if event.button() == Qt.LeftButton:
#             self.endPoint = event.pos()
#             # 进行重新绘制
#             self.update()
#             self.isDrawing = False
#             print(self.endPoint)
#
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     form = Winform()
#     form.show()
#     sys.exit(app.exec_())
#
# import numpy as np
# from scipy.misc import comb
#
# def bernstein_poly(i, n, t):
#     """
#      The Bernstein polynomial of n, i as a function of t
#     """
#
#     return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
#
#
# def bezier_curve(points, nTimes=1000):
#     """
#        Given a set of control points, return the
#        bezier curve defined by the control points.
#
#        points should be a list of lists, or list of tuples
#        such as [ [1,1],
#                  [2,3],
#                  [4,5], ..[Xn, Yn] ]
#         nTimes is the number of time steps, defaults to 1000
#
#         See http://processingjs.nihongoresources.com/bezierinfo/
#     """
#
#     nPoints = len(points)
#     xPoints = np.array([p[0] for p in points])
#     yPoints = np.array([p[1] for p in points])
#
#     t = np.linspace(0.0, 1.0, nTimes)
#
#     polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
#
#     xvals = np.dot(xPoints, polynomial_array)
#     yvals = np.dot(yPoints, polynomial_array)
#
#     return xvals, yvals
#
#
# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#
#     nPoints = 4
#     points = np.random.rand(nPoints,2)*200
#     xpoints = [p[0] for p in points]
#     ypoints = [p[1] for p in points]
#
#     xvals, yvals = bezier_curve(points, nTimes=1000)
#     plt.plot(xvals, yvals)
#     plt.plot(xpoints, ypoints, "ro")
#     for nr in range(len(points)):
#         plt.text(points[nr][0], points[nr][1], nr)
#
#     plt.show()

import cv2
from utils.Functions import splitConnectedComponents

path = "quan.png"

img = cv2.imread(path, 0)

_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

radicals = splitConnectedComponents(img)

print("number of radicals: %d" % len(radicals))

for id, ra in enumerate(radicals):
    print(ra.shape)
    cv2.imshow("id_"+str(id), ra)

img = cv2.resize(img, (int(img.shape[0]/2.), int(img.shape[1]/2.)))
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

cv2.imwrite("quan1.png", img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
