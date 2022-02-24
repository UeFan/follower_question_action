import os

import json
import cv2
import numpy as np
import pickle
import pandas as pd
import scipy.spatial as spt
import random


def get_direction(start,end):
    vec=np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45
    #          270
    if vec[1] > 0: # lng is postive
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle+90)%360
    return _angle


def gps_to_img_coords(gps):
    return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))


def img_to_gps_coords(img_c):
    return np.array([gps_top_right[0] - lat_ratio * img_c[1], gps_botm_left[1] + lat_ratio * img_c[0]])


root_folder_path = '/Users/fanyue/xview/'

new_data = json.load(open(os.path.join(root_folder_path, 'all_data.json')))
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

def click_and_draw(event, x, y, flags, param):
    # grab references to the global variables
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_RBUTTONDOWN:
        to_remove = []
        for i in range(len(attention_list)):
            if np.linalg.norm(np.array(list(attention_list[i][0])) - np.array([x,y]) ) < attention_list[i][1]:
                to_remove.append( attention_list[i])

        if to_remove != []:
            for k in to_remove:
                attention_list.remove(k)
            im_resized = autoAdjustments_with_convertScaleAbs(im_resized_ori)

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
            for i in range(len(attention_list)):
                cv2.circle(im_resized, (
                    int(attention_list[i][0][0]),
                    int(attention_list[i][0][1])
                ),
                           color=(0, 255, 0), radius=attention_list[i][1],
                           thickness=4 + int(size_boundary[0] / 400))
            cv2.imshow('navigation viewer', im_resized)

    if event == cv2.EVENT_LBUTTONDOWN:
        # print(angle, _angle (y/height)*(corners[3][1]-corners[0][1])*np.sin(-angle/ 180 * 3.14159),(x/width)*(corners[1][0]-corners[0][0])*np.sin(angle/ 180 * 3.14159), (x/width)*(corners[1][0]-corners[0][0])*np.cos(angle/ 180 * 3.14159), (y/height)*(corners[3][1]-corners[0][1])*np.cos(angle/ 180 * 3.14159))
        # ((corners[1][0]-corners[0][0]), (corners[3][1]-corners[0][1])), corners[0],(int((x/width)*(corners[1][0]-corners[0][0])*np.cos(angle)+corners[0][0]), int((y/height)*(corners[3][1]-corners[0][1])*np.cos(angle)+corners[0][1]))

        im_resized = autoAdjustments_with_convertScaleAbs(im_resized_ori)

        attention_r = 0
        for i in range(len(pos_list)):
            view_width = np.linalg.norm(np.array(pos_list[i][0]) - np.array(pos_list[i][1]))
            if np.linalg.norm(np.mean(pos_list[i], axis =0) - np.array([x,y]) ) < view_width/2 * 1.5:
                attention_r = int(view_width / 10)
                break
        attention_list.append([[int(x), int(y)], attention_r])
        # cv2.circle(im_resized,
        #            (int(x), int(y)),
        #            attention_r, (0, 255, 0),
        #            2)
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
                   color=(0, 0, 255), radius=4 + int(size_boundary[0] / 400), thickness=4 + int(size_boundary[0] / 400))
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
        for i in range(len(attention_list)):
            cv2.circle(im_resized, (
                int(attention_list[i][0][0]),
                int(attention_list[i][0][1])
            ),
                       color=(0, 255, 0), radius=attention_list[i][1],
                       thickness=4 + int(size_boundary[0] / 400))






        cv2.imshow('navigation viewer', im_resized)


cv2.setMouseCallback("navigation viewer", click_and_draw)
def autoAdjustments_with_convertScaleAbs(img):
    alow = img.min()-30
    ahigh = img.max()
    amax = 255
    amin = 0

    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return new_img
count_i = 0
for q in range(0 ,len(name_list)):
    iii = name_list[q]
    for ii in sub_traj_id_to_idx[iii].keys():
        pos_list = []
        attention_list = []
        for i in range(1, np.max(list(sub_traj_id_to_idx[iii][ii].keys())) + 1):
            p_dic = new_data[sub_traj_id_to_idx[iii][ii][i]]
            lng_ratio = p_dic['lng_ratio']
            lat_ratio = p_dic['lat_ratio']
            gps_botm_left = p_dic['gps_botm_left']
            gps_top_right = p_dic['gps_top_right']
            attention_list += p_dic['attention_list']
        for i in range(len(attention_list)):
            attention_list[i][0] = gps_to_img_coords(attention_list[i][0])


        for i in range(1, np.max(list(sub_traj_id_to_idx[iii][ii].keys())) + 1):
            p_dic = new_data[sub_traj_id_to_idx[iii][ii][i]]
            pos_list = p_dic['gt_path_corners']

            lng_ratio = p_dic['lng_ratio']
            lat_ratio = p_dic['lat_ratio']
            gps_botm_left = p_dic['gps_botm_left']
            gps_top_right = p_dic['gps_top_right']

            dialog = p_dic['instructions']
        
            for i in range(len(pos_list)):
                pos_list[i] = [gps_to_img_coords(pos_list[i][j]) for j in range(4)]


            starting_coord = np.mean(pos_list[0], axis=0)

            count_i += 1
            print('# ', count_i)
            print('\n q:', q, 'iii: ', iii, 'ii: ', ii)
            dialog = dialog.replace('[', '\n[')
            print(dialog)
            print()

            im_full_map = cv2.imread(root_folder_path + 'train_images/' + str(iii)+ ".tif", 1)
            im_resized_ori = cv2.resize(im_full_map, (int(im_full_map.shape[1] * lng_ratio / lat_ratio), im_full_map.shape[0]),
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
            angle = round(get_direction(center_coord, (np.array(pos_list[-1][0]) + np.array(pos_list[-1][1])) / 2)) % 360

            cv2.line(im_resized,
                     (
                         int(center_coord[0] - compass_size_edge * 0.35 * np.sin((angle + 135) / 180 * 3.14159)),
                         int(center_coord[1] + compass_size_edge * 0.35 * np.cos((angle + 135) / 180 * 3.14159))
                     ),
                     (int(center_coord[0]-  compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)), int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                     (0, 0, 255), 2 + int(size_boundary[0] / 400))

            cv2.line(im_resized,
                     (
                         int(center_coord[0] - compass_size_edge * 0.35 * np.sin((angle + 225) / 180 * 3.14159)),
                         int(center_coord[1] + compass_size_edge * 0.35 * np.cos((angle + 225) / 180 * 3.14159))
                     ),
                     (int(center_coord[0]-  compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)), int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                     (0, 0, 255), 2 + int(size_boundary[0] / 400))

            cv2.line(im_resized,
                     (
                         int(center_coord[0]),
                         int(center_coord[1])
                     ),
                     (int(center_coord[0]-  compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)), int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                     (0, 0, 255), 2 + int(size_boundary[0] / 400))

            cv2.circle(im_resized, (
                         int(center_coord[0]),
                         int(center_coord[1])
                     ),
                     color=(0, 0, 255), radius = 4 + int(size_boundary[0] / 400), thickness = 4 + int(size_boundary[0] / 400))
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
            for i in range(len(attention_list)):
                cv2.circle(im_resized, (
                    int(attention_list[i][0][0]),
                    int(attention_list[i][0][1])
                ),
                           color=(0, 255, 0), radius=attention_list[i][1],
                           thickness=4 + int(size_boundary[0] / 400))

            cv2.imshow('navigation viewer',
                        im_resized)


            k = cv2.waitKey(0)

        gps_attention = []
        for k in attention_list:

            gps_attention.append([img_to_gps_coords(k[0]), k[1]])
            # print(gps_attention[-1])

        pickle.dump(gps_attention, open(root_folder_path + str(iii) + '_' + str(ii) + '.pickle', 'wb'))





