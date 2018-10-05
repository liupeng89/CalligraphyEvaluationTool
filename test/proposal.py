import cv2
import numpy

from utils.Functions import getCenterOfGravity, getSingleMaxBoundingBoxOfImage, \
    getAllMiniBoundingBoxesOfImage, getSkeletonOfImage, getContourOfImage, getContourImage, removeExtraBranchesOfSkeleton

path = "test_images/yong.png"

img = cv2.imread(path)

img_bit = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, img_bit = cv2.threshold(img_bit, 100, 255, cv2.THRESH_BINARY)

# remove littine noise
boxes = getAllMiniBoundingBoxesOfImage(img_bit)
for box in boxes:
    if box[2] < 50 or box[3] < 50:
        img_bit[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 255

# size and aspect ratio
img_rbg = cv2.cvtColor(img_bit, cv2.COLOR_GRAY2RGB)
img_layout = img_rbg.copy()

x, y, w, h = getSingleMaxBoundingBoxOfImage(img_bit)
img_rbg = cv2.rectangle(img_rbg, (x,y), (x+w, y+h), (0,255,0),2)
img_cog = img_rbg.copy()

cv2.line(img_rbg, (x,y), (x+w, y+h), (0,0,255), 2 )

# cog
cog_x, cog_y = getCenterOfGravity(img_bit)
print(cog_x, cog_y)

img_cog = cv2.circle(img_cog, (cog_x, cog_y), 2,(0,0,255), 10)

# skeleton
img_skel = cv2.imread("skel.png", 0)
_, img_skel = cv2.threshold(img_skel, 220, 255, cv2.THRESH_BINARY)

img_stroke = cv2.imread("stroke_2.png", 0)
_, img_stroke = cv2.threshold(img_stroke, 127, 255, cv2.THRESH_BINARY)

contours = getContourOfImage(img_stroke)

for y in range(contours.shape[0]):
    for x in range(contours.shape[1]):
        if contours[y][x] != 255:
            img_skel[y][x] = contours[y][x]

# stroke features
img_sk_rgb = cv2.cvtColor(img_stroke, cv2.COLOR_GRAY2RGB)
img_sk_mini = img_sk_rgb.copy()
img_sk_ar = img_sk_rgb.copy()
# img_sk_cog = img_sk_rgb.copy()

x, y, w, h = getSingleMaxBoundingBoxOfImage(img_stroke)
cv2.rectangle(img_sk_mini, (x,y), (x+w, y+h), (0, 255, 0), 2)
img_sk_cog = img_sk_mini.copy()

cv2.line(img_sk_mini, (x,y), (x+w,y+h), (0,0,255), 2)

# cog
cog_x, cog_y = getCenterOfGravity(img_stroke)
cv2.circle(img_sk_cog, (cog_x, cog_y), 2,(0,0,255), 10)


# cv2.imshow("src", img_sk_rgb)
# cv2.imshow("mini", img_sk_mini)
# cv2.imshow("cog", img_sk_cog)
#
# cv2.imwrite("mini.png", img_sk_mini)
# cv2.imwrite("stroke_cog.png", img_sk_cog)

# stroke layout
strokes = []
for i in range(1, 6):
    sk_name = "stroke_" + str(i) + ".png"
    stroke = cv2.imread(sk_name, 0)
    # strokes.append(stroke)
    _, stroke = cv2.threshold(stroke, 127, 255, cv2.THRESH_BINARY)
    sx,sy,sw,sh = getSingleMaxBoundingBoxOfImage(stroke)
    color = (0, 0, 255)
    if i % 3 == 2:
        color = (0 ,255, 0)
    elif i % 3 == 0:
        color = (255, 0, 0)
    cv2.rectangle(img_layout, (sx, sy), (sx+sw, sy+sh), color, 2)

print(len(strokes))

# cv2.imshow("img_layout", img_layout)
# cv2.imwrite("img_layout.png", img_layout)


# cv2.imshow("src", img)
# cv2.imshow("big", img_bit)
# cv2.imshow("rgb", img_rbg)
# cv2.imshow("cog", img_cog)
#
# cv2.imshow("skel", img_skel)
#
# cv2.imwrite("src.png", img)
# cv2.imwrite("bit.png", img_bit)
# cv2.imwrite("rgb.png", img_rbg)
# cv2.imwrite("cog.png", img_cog)
# cv2.imwrite("skel.png", img_skel)

edge = getContourImage(img_bit)

skel = getSkeletonOfImage(img_bit)
skel = removeExtraBranchesOfSkeleton(skel, distance_threshod=50)

cv2.imshow("edge", edge)
cv2.imshow("skel", skel)
cv2.imwrite("edge.png", edge)
cv2.imwrite("skel.png", skel)


cv2.waitKey(0)
cv2.destroyAllWindows()