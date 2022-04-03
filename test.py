import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # Polygon

line1 = [0,0,0,2,2,2,2,0, 0,0]  # One-dimensional array representation of the coordinates of the four points of the quadrilateral, [x,y,x,y....]
a = np.array(line1).reshape(5, 2)  # quadrilateral two-dimensional coordinate representation
poly1 = Polygon(
    a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
print(Polygon(a).convex_hull)  # you can print to see if this is the case

line2 = [-1, -1, -4, -1, -4, -4, -1, -4]
b = np.array(line2).reshape(4, 2)
poly2 = Polygon(b).convex_hull
print(Polygon(b).convex_hull)

union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
# print(union_poly)
print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
    iou = 0
else:
    try:
        inter_area = poly1.intersection(poly2).area  # intersection area
        print(inter_area)
        # union_area = poly1.area + poly2.area - inter_area
        union_area = MultiPoint(union_poly).convex_hull.area
        print(union_area)
        if union_area == 0:
            iou = 0
        # iou = float(inter_area)/(union_area-inter_area)  #wrong
        iou = float(inter_area) / union_area
        # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
        # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
        # The second one: intersection/merge (common way to calculate IOU of rectangular box)
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
        iou = 0

print(a)

print(iou)


