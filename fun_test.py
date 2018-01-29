from  scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


def getTrangleArea(x0,y0,x1,y1,x2,y2):
    area = 0.0

    return area


def getPolygonArea(points):
    if points is None:
        return 0.0
    area = 0.0
    i = j = len(points)-1

    for i in range(len(points)):
        area += (points[j][1] + points[i][1]) * (points[j][0] - points[i][0])
        j = i

    return area * 0.5

points = [(0,0),(1,0),(1,1),(2,2),(0,2),(0,1)]
area = getPolygonArea(points)
print("area: %f" % area)