## The procedure of Calligraphy contours smooth of character:

    1. Load original image of RGB;

    2. Convert RGB image to grayscale image;

    3. Convert grayscale image to binary image;

    4. Separate character into several connected components and process each component one by one;

    5. Corner detect (using Harries corner detector) to get points of corner regions;

    6. Determine the center points of these corner regions;

    7. Get contours of component; if there are holes, the number of contours are larger than 1, otherwise it is 1;

    8. Process contours to get closed and 1-pixel width contours;

    9. Find valid corner points of contours that closed to the center points of corner regions;

    10. Separate contours into sub-contours based on the corner points;

    11. Cubic Bezier curve fit all sub-contours with max-error threshold;

    12. Merge sub-contours together;

    13. Merge all smoothed components;

    14. Return smoothed image.