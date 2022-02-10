import os

import json
import cv2
import numpy as np
import pickle
import pandas as pd
import scipy.spatial as spt
import  random
def polygon_area(points):
    hull = spt.ConvexHull(points=points)
    return hull.area


def gps_to_img_coords(gps):
    return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))


def img_to_gps_coords(img_c):
    return np.array([gps_top_right[0] - lat_ratio * img_c[1], gps_botm_left[1] + lat_ratio * img_c[0]])


dialog_phase = 1
root_folder_path = '/Users/fanyue/xview/'

result_csv = '0_AMT_full_dataset.csv'
df = pd.read_csv(root_folder_path + result_csv)
name_list = []
for i in df['task_image_name']:
    name_list.append(i.replace('image_sample_', '').replace('.jpg',''))

# open a opencv window and display the initial view
cv2.namedWindow('navigation viewer')

for iii in range(0 ,len(df['task_image_name'])):

    if not os.path.exists(root_folder_path + df['task_image_name'][iii]):
        print(iii, '!!!: ', df['task_image_name'][iii])
        continue
    if os.path.exists(
            root_folder_path + name_list[iii].replace('/' + str(0) + '/', '/' + str(dialog_phase) + '/') + "_" + str(
                    dialog_phase) + ".pickle") \
        and os.path.exists(
            root_folder_path + name_list[iii].replace('/' + str(0) + '/',
                                                      '/' + str(dialog_phase) + '/image_sample_') + "_" + str(
                    dialog_phase) + ".jpg"):
        print()
        print(root_folder_path + '/'.join(name_list[iii].split('/')[:-1]) + '/'+name_list[iii].split('/')[1] + ".tif.pickle")

        p_dic = pickle.load(
            open(root_folder_path + '/'.join(name_list[iii].split('/')[:-1]) + '/'+name_list[iii].split('/')[1] + ".tif.pickle",
                 "rb"))

        extracted_landmarks = p_dic['extracted_landmarks']
        extracted_xview_landmarks = p_dic['extracted_xview_landmarks']

        print(root_folder_path + name_list[iii].replace('/' + str(dialog_phase - 1) + '/',
                                                                           '/' + str(dialog_phase) + '/') + '_' + str(dialog_phase) + ".pickle")
        p_dic = pickle.load(open(root_folder_path + name_list[iii].replace('/' + str(dialog_phase - 1) + '/',
                                                                           '/' + str(dialog_phase) + '/') + '_' + str(dialog_phase) + ".pickle",
                                 "rb"))


        print(adsfa)
        pos_list = p_dic['pos_list']
        length_of_traj = p_dic['length_of_traj']
        if type(length_of_traj) != type([]):
            length_of_traj = [length_of_traj]
        angle_list = p_dic['angle_list']
        action_list = p_dic['action_list']
        attention_list = p_dic['attention_list']
        step_change_of_view_zoom = p_dic['step_change_of_view_zoom']
        lng_ratio = p_dic['lng_ratio']
        lat_ratio = p_dic['lat_ratio']
        gps_botm_left = p_dic['gps_botm_left']
        gps_top_right = p_dic['gps_top_right']
        dialog = p_dic['dialog']
        question = p_dic['question']
        approve = p_dic['Approve']
        if approve != 'X':
            continue
        if 'rej' in dialog or 'rej' in question or len(angle_list) == 0:
            p_dic['approve'] = 'rejected by our manual checking.'
            pickle.dump(p_dic, open(root_folder_path + name_list[iii].replace('/' + str(dialog_phase - 1) + '/',
                                                                       '/' + str(dialog_phase) + '/') + '_' + str(
                dialog_phase) + ".pickle",
                             "wb"))
            continue

        _zoom_speed = 8
        print(root_folder_path + name_list[iii].replace('/' + str(dialog_phase) + '/', '/0/').replace(
                "_" + str(dialog_phase), '') + ".pickle")
        p_dic = pickle.load(open(
            root_folder_path + name_list[iii].replace('/' + str(dialog_phase) + '/', '/0/').replace(
                "_" + str(dialog_phase), '') + ".pickle", "rb"))

        destination_index = p_dic['destination_index']
        destination_gps = extracted_xview_landmarks[int(destination_index)][2]

        starting_coord = np.mean(pos_list[0], axis=0)


        print(dialog)
        print(question)
        angle = angle_list[-1]

        path = root_folder_path + name_list[iii].replace('/' + str(0) + '/',
                                                      '/' + str(dialog_phase) + '/image_sample_') + "_" + str(
                                                        dialog_phase) + ".jpg"

        im_full_map = cv2.imread(root_folder_path + 'train_images/' + name_list[iii].split('/')[1]+ ".tif", 1)
        im_resized_copy = cv2.resize(im_full_map, (int(im_full_map.shape[1] * lng_ratio / lat_ratio), im_full_map.shape[0]),
                                interpolation=cv2.INTER_AREA)  # ratio_all = lat_ratio
        cv2.imshow('viewer', im_resized_copy)




        ############## optimize the traj ####################
        # p_dic = pickle.load(open(root_folder_path + name_list[iii].replace('/' + str(dialog_phase - 1) + '/',
        #                                                                    '/' + str(dialog_phase) + '/') + '_' + str(
        #     dialog_phase) + ".pickle",
        #                          "rb"))
        # pos_list = p_dic['pos_list']
        if type(pos_list[0]) != type(np.array([])):
            continue
        print(length_of_traj)
        if len(length_of_traj)>1:
            new_pos_list = [pos_list[:length_of_traj[-2]]]
        else:
            new_pos_list = [pos_list[0]]
        while (1):
            for p_i in range(len(pos_list[::-1])):
                p = pos_list[::-1][p_i]

                if np.linalg.norm(img_to_gps_coords(np.mean(new_pos_list[-1], axis = 0)) - img_to_gps_coords(np.mean(p, axis = 0))) < \
                    np.linalg.norm(img_to_gps_coords(np.mean(new_pos_list[-1], axis = 0)) - img_to_gps_coords( new_pos_list[-1][0] )) * 0.4:
                    new_pos_list.append(p)
                    print(p_i)
                    break
            if np.linalg.norm(
                img_to_gps_coords(np.mean(new_pos_list[-1], axis=0)) - img_to_gps_coords(np.mean(pos_list[-1], axis=0))) < 0.00001:
                break



        p_dic = pickle.load(open(root_folder_path + name_list[iii].replace('/' + str(dialog_phase - 1) + '/',
                                                                           '/' + str(dialog_phase) + '/') + '_' + str(
            dialog_phase) + ".pickle",
                                 "rb"))
        pos_list= new_pos_list
        p_dic ['pos_list'] = pos_list
        length_of_traj[-1] = len(pos_list)
        p_dic['length_of_traj'] = length_of_traj
        print(length_of_traj)

        pickle.dump(p_dic, open(root_folder_path + name_list[iii].replace('/' + str(dialog_phase - 1) + '/',
                                                                           '/' + str(dialog_phase) + '/') + '_' + str(dialog_phase) + ".pickle",
                                 'wb'))



        __coords = []
        __coords.append(pos_list[0])
        _i = 0
        for i in pos_list:
            _i += 1
            mean_im_coords = np.mean(i, axis=0)
            # print(mean_im_coords)
            if (__coords != []):
                cv2.line(im_resized_copy, (int(mean_im_coords[0]), int(mean_im_coords[1])),
                         np.array(np.mean(__coords[-1], axis=0), dtype=np.int32), (255, 0, 255), 4)
            __coords.append([mean_im_coords])

            if _i == len(pos_list):
                cv2.drawContours(im_resized_copy, [np.array(
                    [[int(i[0][0]), int(i[0][1])], [int(i[1][0]), int(i[1][1])], [int(i[2][0]), int(i[2][1])],
                     [int(i[3][0]), int(i[3][1])]])], 0, (255, 255, 255), 1)
                __coords.append(i)



        _gps = np.vstack((np.vstack(__coords), \
                          np.array(
                              [list(starting_coord)] + [list(gps_to_img_coords((destination_gps)))] + [
                                  gps_to_img_coords(i)
                                  for i in
                                  extracted_xview_landmarks[
                                      destination_index][
                                      1]])
                          ))
        im_min_boundary = _gps.min(axis=0)
        im_max_boundary = _gps.max(axis=0)


        size_boundary = np.array(im_max_boundary) - np.array(im_min_boundary) + np.array([100, 100])
        center_coord = np.mean(pos_list[-1], axis=0)

        im_min_boundary[0] = int(max(center_coord[0] - size_boundary[0], 0))
        im_min_boundary[1] = int(max(center_coord[1] - size_boundary[1], 0))

        im_max_boundary[0] = int(min(center_coord[0] + size_boundary[0], im_resized_copy.shape[1]))
        im_max_boundary[1] = int(min(center_coord[1] + size_boundary[1], im_resized_copy.shape[0]))

        compass_size = int(np.linalg.norm(np.array(pos_list[-1][0]) - np.array(pos_list[-1][2])) / 20) + 80
        compass_size_center = int(compass_size * 0.45)

        compass_size_edge = int(compass_size * 0.75)

        cv2.line(im_resized_copy,
                 (
                     int(center_coord[0] - compass_size_edge * 0.35 * np.sin((angle + 135) / 180 * 3.14159)),
                     int(center_coord[1] + compass_size_edge * 0.35 * np.cos((angle + 135) / 180 * 3.14159))
                 ),
                 (int(center_coord[0]-  compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)), int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                 (0, 0, 255), 2 + int(size_boundary[0] / 400))

        cv2.line(im_resized_copy,
                 (
                     int(center_coord[0] - compass_size_edge * 0.35 * np.sin((angle + 225) / 180 * 3.14159)),
                     int(center_coord[1] + compass_size_edge * 0.35 * np.cos((angle + 225) / 180 * 3.14159))
                 ),
                 (int(center_coord[0]-  compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)), int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                 (0, 0, 255), 2 + int(size_boundary[0] / 400))

        cv2.line(im_resized_copy,
                 (
                     int(center_coord[0]),
                     int(center_coord[1])
                 ),
                 (int(center_coord[0]-  compass_size_edge * 0.7 * np.sin((angle + 180) / 180 * 3.14159)), int(center_coord[1] + compass_size_edge * 0.7 * np.cos((angle + 180) / 180 * 3.14159))),
                 (0, 0, 255), 2 + int(size_boundary[0] / 400))

        cv2.circle(im_resized_copy, (
                     int(center_coord[0]),
                     int(center_coord[1])
                 ),
                 color=(0, 0, 255), radius = 4 + int(size_boundary[0] / 400), thickness = 4 + int(size_boundary[0] / 400))
        cv2.putText(im_resized_copy, 'Forward direction', (
            int(center_coord[0] - compass_size_edge * np.sin((angle + 180) / 180 * 3.14159)),
            int(center_coord[1] + compass_size_edge * np.cos((angle + 180) / 180 * 3.14159))
        ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.1 + size_boundary[0] / 1200, (0, 0, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)

        cv2.putText(im_resized_copy, 'current view area', np.array(np.min(pos_list[-1], axis=0), dtype=np.int32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 + size_boundary[0] / 1600, (255, 255, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)

        # print(im_min_boundary[1],im_max_boundary[1], im_min_boundary[0],im_max_boundary[0])

        cv2.rectangle(im_resized_copy, gps_to_img_coords(extracted_xview_landmarks[int(destination_index)][1][0]),
                      gps_to_img_coords(extracted_xview_landmarks[int(destination_index)][1][2]), (255, 0, 255), 2)

        cv2.putText(im_resized_copy, 'Destination', np.array(gps_to_img_coords(destination_gps)) + np.array(
            [int(polygon_area(extracted_xview_landmarks[int(destination_index)][1]) * 30000),
             int(polygon_area(extracted_xview_landmarks[int(destination_index)][1]) * 30000)]), \
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 + size_boundary[0] / 800, (255, 0, 255), 1 + int(size_boundary[0] / 300), cv2.LINE_AA)
        __coords = []
        im_resized = im_resized_copy.copy()
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
                       color=(0, 0, 0), radius=attention_list[i][1],
                       thickness=4 + int(size_boundary[0] / 400))

        cv2.imshow('viewer_1',
                    im_resized[int(im_min_boundary[1]):int(im_max_boundary[1]),
                    int(im_min_boundary[0]):int(im_max_boundary[0])])

        im_full_map = cv2.imread(path, 1)
        cv2.imshow('viewer', im_full_map)

        cv2.imwrite(path, im_resized_copy[int(im_min_boundary[1]):int(im_max_boundary[1]),
                    int(im_min_boundary[0]):int(im_max_boundary[0])])

        k = cv2.waitKey(0)






