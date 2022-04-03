import os

import json
import cv2
import numpy as np
import pickle
import pandas as pd
import scipy.spatial as spt
import random
import shapely
import shapely.geometry
from shapely.geometry import Polygon, MultiPoint

def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45
    #          270
    if vec[1] > 0:  # lng is postive
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle + 90) % 360
    return _angle


def gps_to_img_coords(gps):
    return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))


def img_to_gps_coords(img_c):
    return np.array([gps_top_right[0] - lat_ratio * img_c[1], gps_botm_left[1] + lat_ratio * img_c[0]])


root_folder_path = '/Users/fanyue/xview/'

new_data = json.load(open(os.path.join(root_folder_path, 'full_dataset/test_unseen_anno.json')))
sub_traj_id_to_idx = {}

for i in range(len(new_data)):
    item = new_data[i]

    # print([int(item['map_name']), int(item['route_index'].split('_')[0]), int(item['route_index'].split('_')[1])])
    sub_traj_id_to_idx[int(item['map_name'])] = sub_traj_id_to_idx.get(int(item['map_name']), {})
    sub_traj_id_to_idx[int(item['map_name'])][int(item['route_index'].split('_')[0])] = \
        sub_traj_id_to_idx[int(item['map_name'])].get(int(item['route_index'].split('_')[0]), {})

    sub_traj_id_to_idx[int(item['map_name'])][int(item['route_index'].split('_')[0])][
        int(item['route_index'].split('_')[1])] = i

name_list = list(sub_traj_id_to_idx.keys())

# open a opencv window and display the initial view
cv2.namedWindow('navigation viewer')


def autoAdjustments_with_convertScaleAbs(img):
    alow = img.min() - 30
    ahigh = img.max()
    amax = 255
    amin = 0

    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return new_img

def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
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
    return iou


count_i = 0
for q in range(0, len(name_list)):
    iii = name_list[q]
    # if iii != 1051:
    #     continue
    for ii in sub_traj_id_to_idx[iii].keys():
        pos_list = []
        attention_list = []
        for i in range(1, np.max(
                list(sub_traj_id_to_idx[iii][ii].keys())) + 1):  # starts from idx 1 because the sub-traj starts from 1
            p_dic = new_data[sub_traj_id_to_idx[iii][ii][i]]
            lng_ratio = p_dic['lng_ratio']
            lat_ratio = p_dic['lat_ratio']
            gps_botm_left = p_dic['gps_botm_left']
            gps_top_right = p_dic['gps_top_right']
            attention_list += p_dic['attention_list']

        des = np.array(p_dic['destination'])
        mean_des = np.mean(des, axis=0)
        best_width = max(max(np.linalg.norm(des[0] - des[1]), np.linalg.norm(des[2] - des[1])), 40 / 11.13 / 1e4)

        best_goal_view_area = np.array([[mean_des[0] - best_width / 2, mean_des[1] - best_width / 2],
                                        [mean_des[0] - best_width / 2, mean_des[1] + best_width / 2],
                                        [mean_des[0] + best_width / 2, mean_des[1] + best_width / 2],
                                        [mean_des[0] + best_width / 2, mean_des[1] - best_width / 2]])
        best_goal_view_area = [gps_to_img_coords(best_goal_view_area[j]) for j in range(4)]


        for i in range(len(attention_list)):
            # print(attention_list)
            attention_list[i][0] = gps_to_img_coords(attention_list[i][0])

        for sub_traj_idx in range(1, np.max(list(sub_traj_id_to_idx[iii][ii].keys())) + 1):
            p_dic = new_data[sub_traj_id_to_idx[iii][ii][sub_traj_idx]]
            pos_list = p_dic['gt_path_corners'].copy()

            lng_ratio = p_dic['lng_ratio']
            lat_ratio = p_dic['lat_ratio']
            gps_botm_left = p_dic['gps_botm_left']
            gps_top_right = p_dic['gps_top_right']

            dialog = p_dic['instructions']

            for i in range(len(pos_list)):
                pos_list[i] = [gps_to_img_coords(pos_list[i][j]) for j in range(4)]

            print(pos_list)

            starting_coord = np.mean(pos_list[0], axis=0)

            count_i += 1
            print('# ', count_i)
            print('\n q:', q, 'iii: ', iii, 'ii: ', ii)
            dialog = dialog.replace('[', '\n[')
            print(dialog)
            print()

            im_full_map = cv2.imread(root_folder_path + 'train_images/' + str(iii) + ".tif", 1)
            im_resized_ori = cv2.resize(im_full_map,
                                        (int(im_full_map.shape[1] * lng_ratio / lat_ratio), im_full_map.shape[0]),
                                        interpolation=cv2.INTER_AREA)  # ratio_all = lat_ratio
            # cv2.imshow('viewer', im_resized)

            im_resized = autoAdjustments_with_convertScaleAbs(im_resized_ori)

            #
            # if type(pos_list[0]) != type(np.array([])):
            #     continue

            __coords = []
            __coords.append(pos_list[0])
            _i = 0
            for i in pos_list:
                _i += 1
                mean_im_coords = np.mean(i, axis=0)
                # print(mean_im_coords)
                if (__coords != []):
                    cv2.line(im_resized, (int(mean_im_coords[0]), int(mean_im_coords[1])),
                             np.array(np.mean(__coords[-1], axis=0), dtype=np.int32), (255, 0, 255), 4)
                __coords.append([mean_im_coords])

                if _i == len(pos_list):
                    cv2.drawContours(im_resized, [np.array(
                        [[int(i[0][0]), int(i[0][1])], [int(i[1][0]), int(i[1][1])], [int(i[2][0]), int(i[2][1])],
                         [int(i[3][0]), int(i[3][1])]])], 0, (255, 255, 255), 1)
                    __coords.append(i)

            _gps = np.vstack((np.vstack(__coords), \
                              np.array(
                                  [list(starting_coord)])
                              ))
            im_min_boundary = _gps.min(axis=0)
            im_max_boundary = _gps.max(axis=0)

            size_boundary = np.array(im_max_boundary) - np.array(im_min_boundary) + np.array([100, 100])
            center_coord = np.mean(pos_list[-1], axis=0)

            im_min_boundary[0] = int(max(center_coord[0] - size_boundary[0], 0))
            im_min_boundary[1] = int(max(center_coord[1] - size_boundary[1], 0))

            im_max_boundary[0] = int(min(center_coord[0] + size_boundary[0], im_resized.shape[1]))
            im_max_boundary[1] = int(min(center_coord[1] + size_boundary[1], im_resized.shape[0]))

            compass_size = int(np.linalg.norm(np.array(pos_list[-1][0]) - np.array(pos_list[-1][2])) / 20) + 80
            compass_size_center = int(compass_size * 0.45)

            compass_size_edge = int(compass_size * 0.75)

            # print(pos_list[-1])
            angle = round(
                get_direction(center_coord, (np.array(pos_list[-1][0]) + np.array(pos_list[-1][1])) / 2)) % 360

            cv2.line(im_resized,
                     (
                         int(center_coord[0] - compass_size_edge * 0.35 * np.sin((angle + 135) / 180 * 3.14159)),
                         int(center_coord[1] + compass_size_edge * 0.35 * np.cos((angle + 135) / 180 * 3.14159))
                     ),
                     (int(center_coord[0] - compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)),
                      int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                     (0, 0, 255), 2 + int(size_boundary[0] / 400))

            cv2.line(im_resized,
                     (
                         int(center_coord[0] - compass_size_edge * 0.35 * np.sin((angle + 225) / 180 * 3.14159)),
                         int(center_coord[1] + compass_size_edge * 0.35 * np.cos((angle + 225) / 180 * 3.14159))
                     ),
                     (int(center_coord[0] - compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)),
                      int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                     (0, 0, 255), 2 + int(size_boundary[0] / 400))

            cv2.line(im_resized,
                     (
                         int(center_coord[0]),
                         int(center_coord[1])
                     ),
                     (int(center_coord[0] - compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)),
                      int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                     (0, 0, 255), 2 + int(size_boundary[0] / 400))

            cv2.circle(im_resized, (
                int(center_coord[0]),
                int(center_coord[1])
            ),
                       color=(0, 0, 255), radius=4 + int(size_boundary[0] / 400),
                       thickness=4 + int(size_boundary[0] / 400))
            cv2.putText(im_resized, 'Forward direction', (
                int(center_coord[0] - compass_size_edge * np.sin((angle + 180) / 180 * 3.14159)),
                int(center_coord[1] + compass_size_edge * np.cos((angle + 180) / 180 * 3.14159))
            ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.1 + size_boundary[0] / 1200, (0, 0, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)

            cv2.putText(im_resized, 'current view area', np.array(np.min(pos_list[-1], axis=0), dtype=np.int32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 + size_boundary[0] / 1600, (255, 255, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)

            # print(im_min_boundary[1],im_max_boundary[1], im_min_boundary[0],im_max_boundary[0])

            __coords = []










            # end = False
            # while end == False:
            gps_pos_list = p_dic['gt_path_corners']
            for i in range(len(gps_pos_list)):
                corners = np.array(gps_pos_list[i], dtype="float32")

                iou = compute_iou(corners, gps_pos_list[-1])

                progress = np.float32(iou)

                teacher_a = ['0', '0']


                current_pos = np.mean(corners, axis=0)
                print(current_pos)

                # -------- find teacher altitude --------
                min_dis = 1000
                for j in range(len(gps_pos_list) - 1, -1, -1):
                    gt_pos = np.mean(gps_pos_list[j], axis=0)
                    dis_to_current = np.linalg.norm(gt_pos - current_pos)
                    if dis_to_current + 0.00001 < min_dis:  # 0.00001 is in case there are two gt_path_corner are the same
                        min_dis = dis_to_current
                        closest_step_index = j
                teacher_a[1] = float((np.linalg.norm(np.array(gps_pos_list[closest_step_index][0]) - \
                                                        np.array(gps_pos_list[closest_step_index][1])) \
                                         * 11.13 * 1e4 - 40) / (400 - 40))
                # -------- calculate the progress (iou) --------
                # if end or progress > 0.5:
                #     end = True
                #     teacher_a[0] = np.array([0, 0], dtype=np.float32)
                #     continue


                # -------- find teacher next_pos --------
                goal_corner_center = np.mean(gps_pos_list[-1], axis=0)
                polygon = corners
                shapely_poly = shapely.geometry.Polygon(polygon)
                # in teacher forcing learning, the trajectory will be followed step by step

                line = [np.mean(gps_pos_list[j], axis=0) for j in
                        range(len(gps_pos_list))]
                shapely_line = shapely.geometry.LineString(line)
                if type(shapely_poly.intersection(shapely_line)) == shapely.geometry.linestring.LineString:
                    intersection_line = list(shapely_poly.intersection(shapely_line).coords)
                else:
                    intersection_line = []
                    for line_string in shapely_poly.intersection(shapely_line):
                        intersection_line += list(line_string.coords)

                if intersection_line == []:
                    print(line, closest_step_index)
                    target_point_index = -1
                    line = [current_pos] + [np.mean(gps_pos_list[target_point_index], axis=0)]
                    shapely_line = shapely.geometry.LineString(line)
                    intersection_line = list(shapely_poly.intersection(shapely_line).coords)

                if intersection_line == []:
                    print(line, closest_step_index)

                min_distance = 1
                for x in intersection_line:
                    x = np.array(x)
                    _distance = np.linalg.norm(x - goal_corner_center)
                    if _distance < min_distance:
                        min_distance = _distance
                        teacher_a[0] = x


                _net_next_pos = 1e5 * (teacher_a[0] - current_pos)
                _net_y = np.round(1e5 * ((corners[0] + corners[1]) / 2 - current_pos)).astype(np.int)
                _net_x = np.round(1e5 * ((corners[1] + corners[2]) / 2 - current_pos)).astype(np.int)

                A = np.mat([[_net_x[0], _net_y[0]], [_net_x[1], _net_y[1]]])
                b = np.mat([_net_next_pos[0], _net_next_pos[1]]).T
                r = np.linalg.solve(A, b)

                gt_next_pos_ratio = [r[0, 0], r[1, 0]]

                if max(gt_next_pos_ratio) > 1.1:
                    print(teacher_a[0])

                max_of_gt_next_pos_ratio = max(abs(gt_next_pos_ratio[0]), abs(gt_next_pos_ratio[1]), 1)  # in [-1,1]
                gt_next_pos_ratio[0] /= max_of_gt_next_pos_ratio
                gt_next_pos_ratio[1] /= max_of_gt_next_pos_ratio

                teacher_a[0] = np.array(gt_next_pos_ratio, dtype=np.float32)

                compass_pos = 100
                compass_size = 50
                count_frame = 0
                width = 720
                height = 720

                print(teacher_a)
                print(current_pos)

                dst_pts = np.array([[0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1],
                                    [0, height - 1]], dtype="float32")

                mean_im_coords = np.mean(corners, axis=0)

                # the perspective transformation matrix
                corners = np.array([gps_to_img_coords(j) for j in corners], dtype=np.float32)
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                im_view = cv2.warpPerspective(im_resized_ori, M, (width, height))


                cv2.circle(im_view, (
                    int(teacher_a[0][0] * width/2 +  width/2),
                    int( height/2 - teacher_a[0][1] * height/2)
                    ),
               color=(0, 0, 255), radius=4,
               thickness=4)


                cv2.line(im_view, (compass_pos, compass_pos), (int(compass_pos + 20 * np.sin(-angle / 180 * 3.14159)),
                                                               int(compass_pos - 20 * np.cos(-angle / 180 * 3.14159))),
                         (255, 255, 255), 2)
                cv2.putText(im_view, 'N', (int(compass_pos + compass_size * np.sin(-angle / 180 * 3.14159)),
                                           int(compass_pos - compass_size * np.cos(-angle / 180 * 3.14159))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.line(im_view, (compass_pos, compass_pos), (
                    int(compass_pos + 20 * np.sin((-angle + 90) / 180 * 3.14159)),
                    int(compass_pos - 20 * np.cos((-angle + 90) / 180 * 3.14159))),
                         (255, 255, 255), 2)
                cv2.putText(im_view, 'E',
                            (int(compass_pos + compass_size * np.sin((-angle + 90) / 180 * 3.14159)),
                             int(compass_pos - compass_size * np.cos((-angle + 90) / 180 * 3.14159))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.line(im_view, (compass_pos, compass_pos), (
                    int(compass_pos + 20 * np.sin((-angle + 180) / 180 * 3.14159)),
                    int(compass_pos - 20 * np.cos((-angle + 180) / 180 * 3.14159))),
                         (255, 255, 255), 2)
                cv2.putText(im_view, 'S',
                            (int(compass_pos + compass_size * np.sin((-angle + 180) / 180 * 3.14159)),
                             int(compass_pos - compass_size * np.cos((-angle + 180) / 180 * 3.14159))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.line(im_view, (compass_pos, compass_pos), (
                    int(compass_pos + 20 * np.sin((-angle + 270) / 180 * 3.14159)),
                    int(compass_pos - 20 * np.cos((-angle + 270) / 180 * 3.14159))),
                         (255, 255, 255), 2)
                cv2.putText(im_view, 'W',
                            (int(compass_pos + compass_size * np.sin((-angle + 270) / 180 * 3.14159)),
                             int(compass_pos - compass_size * np.cos((-angle + 270) / 180 * 3.14159))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.line(im_view, (int(width / 2 - 85), int(height / 2)), (int(width / 2 + 85), int(height / 2)),
                         (0, 0, 255), 1)

                cv2.line(im_view, (int(width / 2), int(height / 2 - 85)), (int(width / 2), int(height / 2 + 85)),
                         (0, 0, 255), 1)

                cv2.imshow('navigation viewer', im_view)
                k = cv2.waitKey(0)
            for i in pos_list:
                mean_im_coords = np.mean(i, axis=0)
                # print(mean_im_coords)
                if (__coords != []):
                    cv2.line(im_resized, (int(mean_im_coords[0]), int(mean_im_coords[1])),
                             np.array(np.mean(__coords[-1], axis=0), dtype=np.int32), (255, 0, 255), 4)
                __coords.append([mean_im_coords])

                cv2.drawContours(im_resized, [np.array(
                    [[int(i[0][0]), int(i[0][1])], [int(i[1][0]), int(i[1][1])], [int(i[2][0]), int(i[2][1])],
                     [int(i[3][0]), int(i[3][1])]])], 0, (255, 255, 255), 1)

                cv2.drawContours(im_resized, [np.array(best_goal_view_area, dtype=np.int32)], 0, (0, 255, 0), 2)

            for i in range(len(attention_list)):
                cv2.circle(im_resized, (
                    int(attention_list[i][0][0]),
                    int(attention_list[i][0][1])
                ),
                           color=(0, 255, 0), radius=attention_list[i][1],
                           thickness=4 + int(size_boundary[0] / 400))

            a = np.zeros((1000, im_resized.shape[1], 3), dtype=np.uint8)
            t_p = 0
            for u in dialog.split('\n'):
                if len(u) > 100:
                    for qq in range(0, len(u), 100):
                        t_p += 50
                        cv2.putText(a, u[qq:qq + 100], (
                            0, t_p
                        ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.7, (255, 255, 255), 2)
                else:
                    t_p += 50
                    cv2.putText(a, u, (
                        0, t_p
                    ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.7, (255, 255, 255), 2)
            cv2.imshow('navigation viewer', np.vstack((im_resized, a)))

            k = cv2.waitKey(0)







