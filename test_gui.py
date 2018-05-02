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

# import cv2
# from utils.Functions import splitConnectedComponents
#
# path = "quan1.png"
#
# img = cv2.imread(path, 0)
#
# _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# radicals = splitConnectedComponents(img)
#
# print("number of radicals: %d" % len(radicals))
#
# for id, ra in enumerate(radicals):
#     print(ra.shape)
#     cv2.imshow("id_"+str(id), ra)
#
# img = cv2.resize(img, (int(img.shape[0]/2.), int(img.shape[1]/2.)))
# img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
# cv2.imwrite("quan1.png", img_rgb)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

size = (250, 250)
back_gray = np.ones(size, dtype=np.uint8) * 255
back_rgb = cv2.cvtColor(back_gray, cv2.COLOR_GRAY2RGB)

# add border lines with color red and 2 pixels width
back_rgb[0:2, ] = (0, 0, 255)
back_rgb[0:size[0], 0:2] = (0, 0, 255)
back_rgb[size[0]-2:size[0], ] = (0, 0, 255)
back_rgb[0:size[1], size[1]-2:size[1]] = (0, 0, 255)

type = "net_20"


def image_segment_with_net(image, n):
    """
    Segment RGB image with NxN horizontal lines and vertical lines.
    :param image: RGB image
    :param n:
    :return:
    """
    if image is None:
        print("image segment should not be none!")
        return
    if n == 0:
        return image

    # section range
    w = image.shape[0];
    h = image.shape[1]
    sect_w = int(w / n)
    sect_h = int(h / n)

    if w - sect_w * n > n / 2.0:
        print("w: %d  sect_w*n: %d" % (w, sect_w * n))
        sect_w += 1
    else:
        print("w: %d  sect_w*n: %d" % (w, sect_w * n))
    if h - sect_h * n > n / 2.0:
        print("h: %d sect_h*n: %d" %(h, sect_h * n))
        sect_h += 1

    for i in range(1, n):
        cv2.line(image, (0, i * sect_w), (h - 1, i * sect_w), (0, 0, 255), 1)
        cv2.line(image, (i * sect_h, 0), (i * sect_h, w - 1), (0, 0, 255), 1)

    return image


if type == "mizi":
    # MiZi grid
    cv2.line(back_rgb, (0,0), (size[0]-1, size[1]-1), (0, 0, 255), 1)
    cv2.line(back_rgb, (0, size[1]-1), (size[0]-1, 0), (0, 0, 255), 1)

    midd_w = int(size[0]/2.)
    midd_h = int(size[1]/2.)

    cv2.line(back_rgb, (0, midd_h), (size[0]-1, midd_h), (0, 0, 255), 1)
    cv2.line(back_rgb, (midd_w, 0), (midd_w, size[1]-1), (0, 0, 255), 1)

elif type == "jingzi":
    # JingZi grid
    n = 3
    back_rgb = image_segment_with_net(back_rgb, n)

elif type == "net_20":
    n = 8
    back_rgb = image_segment_with_net(back_rgb, n)


cv2.imshow("gray", back_gray)
cv2.imshow("rgb", back_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()