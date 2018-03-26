import cv2
import math
import numpy as np
from utils.Functions import getContourOfImage, order_points
import matplotlib.pyplot as plt

from scipy import interpolate

def main():

    # load image
    img_path = "../templates/stroke_dan.png"

    img = cv2.imread(img_path, 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # get contour of image
    contour = getContourOfImage(img)
    contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)

    # contor_num = 0
    # for y in range(contour.shape[0]):
    #     for x in range(contour.shape[1]):
    #         if contour[y][x] == 0.0:
    #             contor_num += 1
    #
    # print("number of points in no-order contour: %d" % contor_num)

    # order the contour points
    contour_points_ordered = order_points(contour)
    # contour_points_counter_clockwise = order_points(contour, isClockwise=False)
    print("number of points in ordered contour: %d" % len(contour_points_ordered))
    # print("counter clock: %d" % len(contour_points_counter_clockwise))

    contour_rgb_clock = contour_rgb.copy()
    # contour_rgb_counter_clock = contour_rgb.copy()

    # for id, pt in enumerate(contour_points_ordered):
    #     if id % 100 == 0:
    #         cv2.circle(contour_rgb_clock, pt, 2, (0,255,0),-1)
    #         cv2.putText(contour_rgb_clock, str(id), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)

    # for id, pt in enumerate(contour_points_counter_clockwise):
    #     if id % 100 == 0:
    #         cv2.circle(contour_rgb_counter_clock, pt, 2, (0,255,0),-1)
    #         cv2.putText(contour_rgb_counter_clock, str(id), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)


    # get key points on contour
    corners = cv2.goodFeaturesToTrack(contour, 10, 0.01, 10)
    corners = np.int0(corners)

    index = 0
    corner_points_ = []
    for i in corners:
        x,y = i.ravel()
        pt_ = None
        if (x, y) in contour_points_ordered:
            pt_ = (x, y)
        elif (x, y-1) in contour_points_ordered:
            pt_ = (x, y-1)
        elif (x+1, y-1) in contour_points_ordered:
            pt_ = (x+1, y-1)
        elif (x+1, y) in contour_points_ordered:
            pt_ = (x+1, y)
        elif (x+1, y+1) in contour_points_ordered:
            pt_ = (x+1, y+1)
        elif (x, y+1) in contour_points_ordered:
            pt_ = (x, y+1)
        elif (x-1, y+1) in contour_points_ordered:
            pt_ = (x-1, y+1)
        elif (x-1, y) in contour_points_ordered:
            pt_ = (x-1, y)
        elif (x-1, y-1) in contour_points_ordered:
            pt_ = (x-1, y-1)
        corner_points_.append(pt_)
        cv2.circle(contour_rgb, (pt_[0], pt_[1]), 1, (0, 0, 255), -1)
        cv2.putText(contour_rgb, str(index), (pt_[0], pt_[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2,cv2.LINE_AA)
        index += 1
    print("orignal corner points number: %d" % len(corner_points_))
    # order the corner points in the clockwise direction
    corner_points = []
    index = 0
    for pt in contour_points_ordered:
        if pt in corner_points_:
            corner_points.append(pt)
            cv2.circle(contour_rgb_clock, (pt[0], pt[1]), 3, (255, 0, 0), -1)
            cv2.putText(contour_rgb_clock, str(index), (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
            index += 1
    print("corner points len: %d" % len(corner_points))
    # contour segmentation based on the corner points
    contour_lines = []
    for id in range(len(corner_points)):
        start_point = corner_points[id]
        end_point = None
        if id == len(corner_points) - 1:
            end_point = corner_points[0]
        else:
            end_point = corner_points[id+1]
        # contour segmentation
        contour_segmentation = []
        start_index = contour_points_ordered.index(start_point)
        end_index = contour_points_ordered.index(end_point)

        if start_index <= end_index:
            # normal index
            contour_segmentation = contour_points_ordered[start_index: end_index+1]
        else:
            # end is at
            contour_segmentation = contour_points_ordered[start_index: len(contour_points_ordered)] + \
                                    contour_points_ordered[0: end_index+1]
        contour_lines.append(contour_segmentation)

    print("number of contour segmentation: %d" % len(contour_lines))

    # use different color to show the contour segmentation
    for id in range(len(contour_lines)):
        if id % 3 == 0:
            # red lines
            for pt in contour_lines[id]:
                contour_rgb_clock[pt[1]][pt[0]] = (0, 0, 255)
        elif id % 3 == 1:
            # blue line
            for pt in contour_lines[id]:
                contour_rgb_clock[pt[1]][pt[0]] = (255, 0, 0)
        elif id % 3 == 2:
            # green line
            for pt in contour_lines[id]:
                contour_rgb_clock[pt[1]][pt[0]] = (0, 255, 0)

    # line examples
    line3 = contour_lines[3]
    pf = fitpath(line3, error=2.5)
    print(pf)

    for pt in pf:
        contour_rgb_clock[pt[1]][pt[0]] = (0, 255, 0)



    cv2.imshow("src", img)
    cv2.imshow("contour", contour)
    cv2.imshow("corners", contour_rgb)
    cv2.imshow("contour clock", contour_rgb_clock)
    # cv2.imshow("contour counter clock", contour_rgb_counter_clock)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTanget = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTanget, maxError)


def fitCubic(points, leftTangent, rightTangent, error):
    # Use heuristic if region only has two points in it
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.
        bezCurve = [points[0], points[0]+leftTangent*dist, points[1]+rightTangent*dist, points[1]]
        return [bezCurve]

    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return [bezCurve]

    # If error not too large, try some reparameterization and iteration
    if maxError < error**2:
        for i in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint-1] - points[splitPoint+1])
    beziers += fitCubic(points[:splitPoint+1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers


def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    # compute the A's
    A = np.zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = leftTangent * 3 * (1-u)**2 *u
        A[i][1] = rightTangent * 3 * (1-u) * u**2

    # Create the C and X matrics
    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] += np.dot(A[i][0], A[i][1])
        C[1][1] += np.dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)
        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    # compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = np.linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezCurve[1] = bezCurve[0] + leftTangent * (segLength / 3.)
        bezCurve[2] = bezCurve[3] + rightTangent * (segLength / 3.)
    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bezCurve[1] = bezCurve[0] + leftTangent * alpha_l
        bezCurve[2] = bezCurve[3] + rightTangent * alpha_r

    return bezCurve


def reparameterize(bezier, points, parameters):
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters)]


def newtonRaphsonRootFind(bez, point, u):
    """
        Newton's root finding algorithm calculates f(x)=0 by reiterating
       x_n+1 = x_n - f(x_n)/f'(x_n)
       We are trying to find curve parameter u for some point p that minimizes
       the distance from that point to the curve. Distance point to curve is d=q(u)-p.
       At minimum distance the point is perpendicular to the curve.
       We are solving
       f = q(u)-p * q'(u) = 0
       with
       f' = q'(u) * q'(u) + q(u)-p * q''(u)
       gives
       u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    :param bez:
    :param point:
    :param u:
    :return:
    """
    d = q(bez, u) - point
    numerator = (d * qprime(bez, u)).sum()
    denominator = (qprime(bez, u)**2 + d * qprimeprime(bez, u)).sum()
    if denominator == 0.0:
        return u
    else:
        return u - numerator / denominator

def chordLengthParameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i-1] + np.linalg.norm(points[i] - points[i-1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]
    return u

def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points) / 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = np.linalg.norm(q(bez, u)-point)**2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i

    return maxDist, splitPoint


def normalize(v):
    return v / np.linalg.norm(v)


# evaluates cubic bezier at t, return point
def q(ctrlPoly, t):
    return (1.0-t)**3 * ctrlPoly[0] + 3*(1.0-t)**2 * t * ctrlPoly[1] + 3*(1.0-t)* t**2 * ctrlPoly[2] + t**3 * ctrlPoly[3]


# evaluates cubic bezier first derivative at t, return point
def qprime(ctrlPoly, t):
    return 3*(1.0-t)**2 * (ctrlPoly[1]-ctrlPoly[0]) + 6*(1.0-t) * t * (ctrlPoly[2]-ctrlPoly[1]) + 3*t**2 * (ctrlPoly[3]-ctrlPoly[2])


# evaluates cubic bezier second derivative at t, return point
def qprimeprime(ctrlPoly, t):
    return 6*(1.0-t) * (ctrlPoly[2]-2*ctrlPoly[1]+ctrlPoly[0]) + 6*(t) * (ctrlPoly[3]-2*ctrlPoly[2]+ctrlPoly[1])


TOLERANCE = 10e-6
EPSILON = 10e-12


class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y=None):
        if y is None:
            self.x, self.y = x[0], x[1]
        else:
            self.x, self.y = x, y

    def __repr__(self):
        return 'Point(%r, %r)' % (self.x, self.y)

    def __str__(self):
        return '%G,%G' % (self.x, self.y)

    def __complex__(self):
        return complex(self.x, self.y)

    def __hash__(self):
        return hash(self.__complex__())

    def __bool__(self):
        return bool(self.x or self.y)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.y)
        else:
            return Point(self.x / other, self.y / other)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __len__(self):
        return math.hypot(self.x, self.y)

    def __eq__(self, other):
        try:
            return self.x == other.x and self.y == other.y
        except Exception:
            return False

    def __ne__(self, other):
        try:
            return self.x != other.x or self.y != other.y
        except Exception:
            return True

    add = __add__
    subtract = __sub__
    multiply = __mul__
    divide = __truediv__
    negate = __neg__
    getLength = __len__
    equals = __eq__

    def copy(self):
        return Point(self.x, self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def normalize(self, length=1):
        current = self.__len__()
        scale = length / current if current != 0 else 0
        return Point(self.x * scale, self.y * scale)

    def getDistance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class Segment:

    def __init__(self, *args):
        self.point = Point(0, 0)
        self.handleIn = Point(0, 0)
        self.handleOut = Point(0, 0)
        if len(args) == 1:
            if isinstance(args[0], Segment):
                self.point = args[0].point
                self.handleIn = args[0].handleIn
                self.handleOut = args[0].handleOut
            else:
                self.point = args[0]
        elif len(args) == 2 and isinstance(args[0], (int, float)):
            self.point = Point(*args)
        elif len(args) == 2:
            self.point = args[0]
            self.handleIn = args[1]
        elif len(args) == 3:
            self.point = args[0]
            self.handleIn = args[1]
            self.handleOut = args[2]
        else:
            self.point = Point(args[0], args[1])
            self.handleIn = Point(args[2], args[3])
            self.handleOut = Point(args[4], args[5])

    def __repr__(self):
        return 'Segment(%r, %r, %r)' % (self.point, self.handleIn, self.handleOut)

    def __hash__(self):
        return hash((self.point, self.handleIn, self.handleOut))

    def __bool__(self):
        return bool(self.point or self.handleIn or self.handleOut)

    def getPoint(self):
        return self.point

    def setPoint(self, other):
        self.point = other

    def getHandleIn(self):
        return self.handleIn

    def setHandleIn(self, other):
        self.handleIn = other

    def getHandleOut(self):
        return self.handleOut

    def setHandleOut(self, other):
        self.handleOut = other


class PathFitter:

    def __init__(self, segments, error=2.5):
        self.points = []
        # Copy over points from path and filter out adjacent duplicates.
        l = len(segments)
        prev = None
        for i in range(l):
            point = segments[i].point.copy()
            if prev != point:
                self.points.append(point)
                prev = point
        self.error = error

    def fit(self):
        points = self.points
        length = len(points)
        self.segments = [Segment(points[0])] if length > 0 else []
        if length > 1:
            self.fitCubic(0, length - 1,
                          # Left Tangent
                          points[1].subtract(points[0]).normalize(),
                          # Right Tangent
                          points[length - 2].subtract(points[length - 1]).normalize())
        return self.segments

    # Fit a Bezier curve to a (sub)set of digitized points
    def fitCubic(self, first, last, tan1, tan2):
        #  Use heuristic if region only has two points in it
        if last - first == 1:
            pt1 = self.points[first]
            pt2 = self.points[last]
            dist = pt1.getDistance(pt2) / 3
            self.addCurve([pt1, pt1 + tan1.normalize(dist),
                           pt2 + tan2.normalize(dist), pt2])
            return
        # Parameterize points, and attempt to fit curve
        uPrime = self.chordLengthParameterize(first, last)
        maxError = max(self.error, self.error * self.error)
        # Try 4 iterations
        for i in range(5):
            curve = self.generateBezier(first, last, uPrime, tan1, tan2)
            #  Find max deviation of points to fitted curve
            maxerr, maxind = self.findMaxError(first, last, curve, uPrime)
            if maxerr < self.error:
                self.addCurve(curve)
                return
            split = maxind
            # If error not too large, try reparameterization and iteration
            if maxerr >= maxError:
                break
            self.reparameterize(first, last, uPrime, curve)
            maxError = maxerr
        # Fitting failed -- split at max error point and fit recursively
        V1 = self.points[split - 1].subtract(self.points[split])
        V2 = self.points[split] - self.points[split + 1]
        tanCenter = V1.add(V2).divide(2).normalize()
        self.fitCubic(first, split, tan1, tanCenter)
        self.fitCubic(split, last, tanCenter.negate(), tan2)

    def addCurve(self, curve):
        prev = self.segments[len(self.segments) - 1]
        prev.setHandleOut(curve[1].subtract(curve[0]))
        self.segments.append(
            Segment(curve[3], curve[2].subtract(curve[3])))

    # Use least-squares method to find Bezier control points for region.
    def generateBezier(self, first, last, uPrime, tan1, tan2):
        epsilon = 1e-11
        pt1 = self.points[first]
        pt2 = self.points[last]
        # Create the C and X matrices
        C = [[0, 0], [0, 0]]
        X = [0, 0]

        l = last - first + 1

        for i in range(l):
            u = uPrime[i]
            t = 1 - u
            b = 3 * u * t
            b0 = t * t * t
            b1 = b * t
            b2 = b * u
            b3 = u * u * u
            a1 = tan1.normalize(b1)
            a2 = tan2.normalize(b2)
            tmp = (self.points[first + i]
                   - pt1.multiply(b0 + b1)
                   - pt2.multiply(b2 + b3))
            C[0][0] += a1.dot(a1)
            C[0][1] += a1.dot(a2)
            # C[1][0] += a1.dot(a2)
            C[1][0] = C[0][1]
            C[1][1] += a2.dot(a2)
            X[0] += a1.dot(tmp)
            X[1] += a2.dot(tmp)

        # Compute the determinants of C and X
        detC0C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
        if abs(detC0C1) > epsilon:
            # Kramer's rule
            detC0X = C[0][0] * X[1] - C[1][0] * X[0]
            detXC1 = X[0] * C[1][1] - X[1] * C[0][1]
            # Derive alpha values
            alpha1 = detXC1 / detC0C1
            alpha2 = detC0X / detC0C1
        else:
            # Matrix is under-determined, try assuming alpha1 == alpha2
            c0 = C[0][0] + C[0][1]
            c1 = C[1][0] + C[1][1]
            if abs(c0) > epsilon:
                alpha1 = alpha2 = X[0] / c0
            elif abs(c1) > epsilon:
                alpha1 = alpha2 = X[1] / c1
            else:
                # Handle below
                alpha1 = alpha2 = 0

        # If alpha negative, use the Wu/Barsky heuristic (see text)
        # (if alpha is 0, you get coincident control points that lead to
        # divide by zero in any subsequent NewtonRaphsonRootFind() call.
        segLength = pt2.getDistance(pt1)
        epsilon *= segLength
        if alpha1 < epsilon or alpha2 < epsilon:
            # fall back on standard (probably inaccurate) formula,
            # and subdivide further if needed.
            alpha1 = alpha2 = segLength / 3

        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        return [pt1, pt1.add(tan1.normalize(alpha1)),
                pt2.add(tan2.normalize(alpha2)), pt2]

    # Given set of points and their parameterization, try to find
    # a better parameterization.
    def reparameterize(self, first, last, u, curve):
        for i in range(first, last + 1):
            u[i - first] = self.findRoot(curve, self.points[i], u[i - first])

    # Use Newton-Raphson iteration to find better root.
    def findRoot(self, curve, point, u):
        # Generate control vertices for Q'
        curve1 = [
            curve[i + 1].subtract(curve[i]).multiply(3) for i in range(3)]
        # Generate control vertices for Q''
        curve2 = [
            curve1[i + 1].subtract(curve1[i]).multiply(2) for i in range(2)]
        # Compute Q(u), Q'(u) and Q''(u)
        pt = self.evaluate(3, curve, u)
        pt1 = self.evaluate(2, curve1, u)
        pt2 = self.evaluate(1, curve2, u)
        diff = pt - point
        df = pt1.dot(pt1) + diff.dot(pt2)
        # Compute f(u) / f'(u)
        if abs(df) < TOLERANCE:
            return u
        # u = u - f(u) / f'(u)
        return u - diff.dot(pt1) / df

    # Evaluate a bezier curve at a particular parameter value
    def evaluate(self, degree, curve, t):
        # Copy array
        tmp = curve[:]
        # Triangle computation
        for i in range(1, degree + 1):
            for j in range(degree - i + 1):
                tmp[j] = tmp[j].multiply(1 - t) + tmp[j + 1].multiply(t)
        return tmp[0]

    # Assign parameter values to digitized points
    # using relative distances between points.
    def chordLengthParameterize(self, first, last):
        u = {0: 0}
        print(first, last)
        for i in range(first + 1, last + 1):
            u[i - first] = u[i - first - 1] + \
                self.points[i].getDistance(self.points[i - 1])
        m = last - first
        for i in range(1, m + 1):
            u[i] /= u[m]
        return u

    # Find the maximum squared distance of digitized points to fitted curve.
    def findMaxError(self, first, last, curve, u):
        index = math.floor((last - first + 1) / 2)
        maxDist = 0
        for i in range(first + 1, last):
            P = self.evaluate(3, curve, u[i - first])
            v = P.subtract(self.points[i])
            dist = v.x * v.x + v.y * v.y  # squared
            if dist >= maxDist:
                maxDist = dist
                index = i
        return maxDist, index


def fitpath(pointlist, error):
    return PathFitter(list(map(Segment, map(Point, pointlist))), error).fit()


def fitpathsvg(pointlist, error):
    return pathtosvg(PathFitter(list(map(Segment, map(Point, pointlist))), error).fit())


def pathtosvg(path):
    segs = ['M', str(path[0].point)]
    last = path[0]
    for seg in path[1:]:
        segs.append('C')
        segs.append(str(last.point + last.handleOut))
        segs.append(str(seg.point + seg.handleIn))
        segs.append(str(seg.point))
        last = seg
    return ' '.join(segs)


if __name__ == '__main__':
    main()