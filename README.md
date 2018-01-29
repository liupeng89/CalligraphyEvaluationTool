# CalligraphyEvaluationTool

Those tools are used to comparse the Chinese calligraphy, including the 
global features, components features and stroke features comparisons. 
Because when people appreciate something, they first get a holistic impression(global 
 features and component features) and then go to details(strokes). Those three features 
are both derived from authoritative calligraphy books and professional calligrapher's 
empirical recommendation.

## Image pre-process

The image pre-process mainly contains image cropping, image binary, image 
inverting, de-noising, removeing background, etc. 

#### Calligraphy characters cropping

The cropping tool is used to extract each single character from the Chinese calligraphy 
images.

#### Calligraphy character image pre-process

We obtain the binary images by pre-processing. The mainly procedure includes: 1). Image binary; 
2). Image invering(option); 3). De-noising 4).Background removing, etc.

As the results of image pre-processing, the size of each images are different. We need to 
resize all images with maintaining the aspect of ratio of each original images.

#### Calligraphy characters contours smoothing
After the image pre-processing, the contour of characters is rough, and a lot of “teeth” exist. 
Therefore, we used the SVG tool to smooth the contour of characters. First, we used the Web tool 
of PNG to SVG converting tool (https://convertio.co/png-svg/) or POTRACE to convert the PNG image 
files to the SVG image files. Secondly, we used the Inkscape app(https://inkscape.org/en/) to 
simply the SVG files to reduce the number of control points. The requirement of simplifying is 
minimizing the loss of valid pixels, while ensuring that there are as few control points as possible. 

The automatic SVG simplifying tool is important to instead of manually reducing control points. 

#### Calligraphy character image resizing

The source images have different size with target images. Before the comparison of the calligraphy 
images, we need to resize the source image and the target image. The basic requirement of resizing 
work is maintaining the aspect of ratio of each source and target image. Our approach is that:

1. Get the minimum bounding boxes of source and target images;
2. Each image are resized to square, and the width of square equals the maximum value of the bounding 
box width and height;
3. Scale the new square image to same size;

#### Calligraphy character image shifting
One simplest approach to compare the source image and target image is using the target image to 
cover the source image, and observe the overlap area between the source and target image. In order 
to calculate the overlap area precisely, we need to shift the region of character in image. We 
used maximum coverage rate to determine the offset of shifting image.


## Global features comparison

In the global feature comparison, we mainly discuss the comparison of maximum coverage rate, histogram 
of X-axis and Y-axis, center of gravity, convex hull statistics, etc.

#### Maximum coverage rate

We can obtain the maximum coverage rate via image shifting. The coverage rate defines as:

    CR = (P_valid - P_less - P_over) / P_valid

#### Histogram of X-axis, Y-axis, rotate 45' and 135'.

There are four types of strokes most frequently used in Chinese characters: the horizontal strokes, 
the vertical strokes, the left-falling strokes and the right-falling strokes. We can observe those 
strokes by projecting black pixels on X-axis, Y-axis, rotate 45' and 135' four histograms.

#### Center of Gravity
In physics point of the center of gravity can be used to describe the stability of a rigid body.


#### Convex hull statistics

Convex hull is used to represent the general shape of characters (Fig.). It can represent the 
alignment and stability of characters. If a stroke is far away from the central region of 
the character, the convex hull may have a sharp corner. Thus, the handwriting sample may 
be skew and oblique rather than aesthetically pleasing. The algorithm of detecting the 
convex hull is Graham scan, which finds all vertices of the convex hull ordered along 
its boundary, and uses a stack to detect and remove concavities in the boundary efficiently.




## Component features comparison
Components are the set of strokes, which are connected interiorly but rarely 
overlap with external strokes.  

We can compute the maximum distance, minimum distance, and mean distance between 
each pair of components and divide these distances by the diagonal distances of 
the character’s minimum bounding box for normalization. 

## Stroke features comparison