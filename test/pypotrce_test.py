from __future__ import division, print_function
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, wsvg, svg2paths, smoothed_path, kinks

paths, attributes = svg2paths("../test_images/src_resize.svg")

# for p in paths:
#     print(type(p))
#     print(len(p))


# print(len(paths))
# print(type(paths))
# paths = paths.pop(1)
#
path = paths[0]
# #
path_smooth = smoothed_path(path, maxjointsize=10)
#
wsvg(path, filename="../test_images/path_0.svg")
wsvg(path_smooth, filename="../test_images/path_s.svg")



# seg1 = CubicBezier(300+100j, 100+100j, 200+200j, 200+300j)
# seg2 = Line(200+300j, 250+350j)
# path = Path(seg1, seg2)
#
# from svgpathtools import parse_path
# path_alt = parse_path('M 300 100 C 100 100 200 200 200 300 L 250 350')
#
# print(path)
# print(path_alt)
#
# print(path == path_alt)
#
# print(path.d())
#
# path.append(CubicBezier(250+350j, 275+350j, 250+225j, 200+100j))
# print(path)
#
# path[0] = Line(200+100j, 200+300j)
# print(path)
#
# print('path is contious?', path.iscontinuous())
# print('path is closed?', path.isclosed())
#
# wsvg(path, filename="path.svg")