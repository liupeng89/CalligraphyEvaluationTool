import numpy as np
import matplotlib.pyplot as plt
import cv2


# Function to konw if we have a CCW turn
def RightTurn(p1, p2, p3):
    if (p3[1]-p1[1]) * (p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
        return False
    return True

# main algorithm
def GrahamScan(P):

    P.sort()
    L_upper = [P[0], P[1]]
    # Compute the upper part of the hull
    for i in range(2, len(P)):
        L_upper.append(P[i])
        while len(L_upper) > 2 and not RightTurn(L_upper[-1], L_upper[-2], L_upper[-3]):
            del L_upper[-2]
    L_lower = [P[-1], P[-2]]
    # compute the lower part of the hull
    for i in range(len(P)-3, -1, -1):
        L_lower.append(P[i])
        while len(L_lower) > 2 and not RightTurn(L_lower[-1], L_lower[-2], L_lower[-3]):
            del L_lower[-2]

    del L_lower[0]
    del L_lower[-1]
    L = L_upper + L_lower
    return np.array(L)

def main():

    N = 100

    src_path = "../src_resize.png"
    tag_path = "../tag_resize.png"

    src_img = cv2.imread(src_path, 0)
    tag_img = cv2.imread(tag_path, 0)

    _, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    _, tag_img = cv2.threshold(tag_img, 127, 255, cv2.THRESH_BINARY)

    # src convex hull
    P = []
    for y in range(src_img.shape[0]):
        for x in range(src_img.shape[1]):
            if src_img[y][x] == 0.0:
                P.append((x, src_img.shape[0] - y))

    # P = [(np.random.randint(0, 300), np.random.randint(0, 300)) for i in range(N)]
    L = GrahamScan(P)
    P = np.array(P)

    # plot figure
    plt.figure()
    plt.plot(L[:, 0], L[:, 1], 'b-', picker=1)
    plt.plot([L[-1, 0], L[0, 0]], [L[-1, 1], L[0, 1]], 'b-', picker=1)
    plt.plot(P[:, 0], P[:, 1], '.r')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
