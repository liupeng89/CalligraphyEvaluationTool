import numpy as np
import cv2

from functions.AddBoundingBox import addBoundingBox

img_file = "../characters/src_bing_processed.png"

img = cv2.imread(img_file, 0)

print(img.shape)

rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

WIDTH = img.shape[0]
HEIGHT = img.shape[1]



# moments
im2, contours, hierarchy = cv2.findContours(img, 1, 2)
print("Contours len: %s " % len(contours))

cnt = contours[0]
M = cv2.moments(cnt)
# print(M)

# center of mass
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print('( %d, %d)' % (cy, cx))


# area
area = cv2.contourArea(cnt)

print("Area:", area)

# Bounding box
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    if w > 0.95 * WIDTH and h > 0.95 * HEIGHT:
        continue

    cv2.rectangle(rgb_img, (x, y), (x+w, y+h), (0,255,0), 1)

minx, miny, minw, minh = addBoundingBox(img)
maxx = minx + minw
maxy = miny + minh

cv2.rectangle(rgb_img, (minx, miny), (maxx, maxy), (255, 0, 0), 1)

aspect_ratio = minh / minw * 1.0
print("Aspect ratio: %f \n" % aspect_ratio)


# convex hull


cv2.imshow("image", rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()