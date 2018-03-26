import cv2
import numpy as np

def cubic_bezier_sum(t, w):
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    return w[0] * mt3 + 3 * w[1] * mt2 * t + 3 * w[2] * mt * t2 + w[3] * t3


def draw_cubic_bezier(p1, p2, p3, p4):
    points = []
    t = 0
    while t < 1:
        x = int(cubic_bezier_sum(t, (p1[0], p2[0], p3[0], p4[0])))
        y = int(cubic_bezier_sum(t, (p1[1], p2[1], p3[1], p4[1])))

        points.append((x, y))

        t += 0.01
    return points

points = []
points.append((635, 489))
points.append((612, 489))
points.append((540, 486))
points.append((519, 476))

bezier_points = draw_cubic_bezier(points[0], points[1], points[2], points[3])

print(len(bezier_points))

img_path = "../templates/stroke_dan.png"

img = cv2.imread(img_path, 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

for id in range(len(bezier_points)-1):
    start_pt = bezier_points[id]
    end_pt = bezier_points[id+1]
    cv2.line(img_rgb, start_pt, end_pt, (0, 255, 0))

cv2.imshow("rgb", img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
