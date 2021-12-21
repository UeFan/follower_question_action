import os

import json
import cv2
import numpy as np
import pickle
import pandas as pd
import scipy.spatial as spt
import random

# you will need to download:
# 18 folder
# .csv (tells you to use 18.tif, automatic)
# /Users/fanyue/xview/xView_train.geojson


# input files:
# /Users/fanyue/xview/18/{*.pickle}
# /Users/fanyue/xview/train_images/18.tif
# /Users/fanyue/Downloads/Batch_4582824_batch_results.csv

# output:
# /Users/fanyue/xview/18/{*_1.pickle}
# /Users/fanyue/xview/18/{image_sample_*_1.jpg}

# steps:
# wasd+oi control, press 'esc'
# input: x means reject. y means already found it. otherwise, questions.
# press enter



dialog_phase = 1
root_folder_path = '/Users/fanyue/xview/'
df = pd.read_csv('/Users/fanyue/Downloads/Batch_4594154_batch_results.csv')
short_cut = pd.read_excel('/Users/fanyue/Downloads/Common questions.xlsx', index_col=0, header=0)
# 710m * 400m = 16:9
# 142 * 80
max_view = np.array([400,400])
min_view = np.array([40,40])
# def zoom_in_out(event, x, y, flags, param):b

def polygon_area(points):
    hull = spt.ConvexHull(points=points)
    return hull.area


def get_a_gps_coord_at_distance(a, b):
    # a is a gps coord
    # b is a distance in meter
    # return a gps coord
    return b/11.13/1e4 + a


def gps_to_img_coords(gps):
    return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))




def img_to_gps_coords(img_c):
    return np.array([gps_top_right[0] - lat_ratio * img_c[1], gps_botm_left[1] + lat_ratio * img_c[0]])

def rotation_anticlock(theta, p):
    M = np.array([[np.cos(theta/180*3.14159), -np.sin(theta/180*3.14159)], [np.sin(theta/180*3.14159), np.cos(theta/180*3.14159)]])
    return np.matmul(M, np.array([p[0], p[1]]))

def change_corner(cs, change): # corners = cs
    new_cs = np.zeros((4,2))
    new_cs[0] = cs[0] + (cs[1] - cs[0])/ np.linalg.norm((cs[1] - cs[0])) * change[0][0]
    new_cs[0] += (cs[3] - cs[0])/ np.linalg.norm((cs[3] - cs[0])) * change[0][1]

    new_cs[1] = cs[1] + (cs[1] - cs[0])/ np.linalg.norm((cs[1] - cs[0])) * change[1][0]
    new_cs[1] += (cs[2] - cs[1])/ np.linalg.norm((cs[2] - cs[1])) * change[1][1]

    new_cs[2] = cs[2] + (cs[2] - cs[3])/ np.linalg.norm((cs[2] - cs[3])) * change[2][0]
    new_cs[2] += (cs[2] - cs[1])/ np.linalg.norm((cs[2] - cs[1])) * change[2][1]

    new_cs[3] = cs[3] + (cs[2] - cs[3])/ np.linalg.norm((cs[2] - cs[3])) * change[3][0]
    new_cs[3] += (cs[3] - cs[0])/ np.linalg.norm((cs[3] - cs[0])) * change[3][1]

    return new_cs

# fname = '/Users/fanyue/xview/xView_train.geojson'
#
# with open(fname) as f:
#     data = json.load(f)
#
#


name_list = []
for i in df['Input.task_image_name']:
    name_list.append(i.replace('image_sample_', '').replace('.jpg',''))


# open a opencv window and display the initial view
cv2.namedWindow('navigation viewer')


def click_and_draw(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(angle, _angle (y/height)*(corners[3][1]-corners[0][1])*np.sin(-angle/ 180 * 3.14159),(x/width)*(corners[1][0]-corners[0][0])*np.sin(angle/ 180 * 3.14159), (x/width)*(corners[1][0]-corners[0][0])*np.cos(angle/ 180 * 3.14159), (y/height)*(corners[3][1]-corners[0][1])*np.cos(angle/ 180 * 3.14159))
        # ((corners[1][0]-corners[0][0]), (corners[3][1]-corners[0][1])), corners[0],(int((x/width)*(corners[1][0]-corners[0][0])*np.cos(angle)+corners[0][0]), int((y/height)*(corners[3][1]-corners[0][1])*np.cos(angle)+corners[0][1]))

        cv2.circle(im_resized,
                   (int((x / width) * np.linalg.norm(corners[1] - corners[0]) * np.cos(angle / 180 * 3.14159) - (
                               y / height) * np.linalg.norm(corners[3] - corners[0]) * np.sin(angle / 180 * 3.14159) +
                        corners[0][0]),
                    int((y / height) * np.linalg.norm(corners[3] - corners[0]) * np.cos(angle / 180 * 3.14159) + (
                                x / width) * np.linalg.norm(corners[1] - corners[0]) * np.sin(angle / 180 * 3.14159) +
                        corners[0][1])),
                   int(np.linalg.norm(corners[1] - corners[0]) * 0.09), (0, 255, 0),
                   2)
        im_view = cv2.warpPerspective(im_resized, M, (width, height))

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


cv2.setMouseCallback("navigation viewer", click_and_draw)

for iii in range(31,len(name_list)):


    img_name = name_list[iii].split('/')[1] +'.tif'
    os.system('mkdir ' + root_folder_path + 'full_dataset/' + name_list[iii].split('/')[1] + '/'+str(dialog_phase+1)+'/')
    print('folder created: ',root_folder_path + 'full_dataset/' + name_list[iii].split('/')[1] + '/'+str(dialog_phase+1)+'/')
    print('Load pickle: ', root_folder_path + name_list[iii].replace('/'+str(dialog_phase-1)+'/','/'+ str(dialog_phase) +'/')+".pickle")

    p_dic = pickle.load( open( root_folder_path  + 'full_dataset/' +  name_list[iii].split('/')[0] + '/0/' +img_name+".pickle", "rb" ) )

    extracted_landmarks = p_dic['extracted_landmarks']
    extracted_xview_landmarks=p_dic['extracted_xview_landmarks']

    path = root_folder_path + 'train_images/' + img_name
    im_full_map = cv2.imread(path, 1)



    ### Get the complete language instruction

    # file1 = open(root_folder_path + name_list[iii].split('/')[0] + '/output_sample_' + name_list[iii].split('/')[1] + '.txt', 'r')
    # Lines = file1.readlines()



    ### Get the boundary coords. The gps coords will be lat,long


    # calculate the coordinate for the edge
    # bb_list = []
    # for i in range(len(data['features'])):
    #     if (data['features'][i]['properties']['image_id'] == img_name):
    #         bb_list.append([i, data['features'][i]['properties']['type_id']])
    #
    # _gps_coord = np.array(data['features'][bb_list[6][0]]['geometry']['coordinates'])
    # gps_coord_0 = np.min(_gps_coord, axis=1)[0]
    #
    # _gps_coord = np.array(data['features'][bb_list[-6][0]]['geometry']['coordinates'])
    # gps_coord_1 = np.max(_gps_coord, axis=1)[0]
    # _im_coords = data['features'][bb_list[6][0]]['properties']['bounds_imcoords'].split(',')
    # im_coords_0 = np.array([int(_im_coords[3]), int(_im_coords[0])])
    #
    # _im_coords = data['features'][bb_list[-6][0]]['properties']['bounds_imcoords'].split(',')
    # im_coords_1 = np.array([int(_im_coords[1]), int(_im_coords[2])])
    # lng_ratio = (gps_coord_1[0] - gps_coord_0[0]) / (im_coords_1[1] - im_coords_0[1])
    # lat_ratio = (gps_coord_1[1] - gps_coord_0[1]) / (im_coords_0[0] - im_coords_1[0])
    #
    # gps_botm_left = [gps_coord_0[1] - lat_ratio * (im.shape[0] - 1 - im_coords_0[0]),
    #                  gps_coord_0[0] - lng_ratio * (im_coords_0[1] - 0)]
    #
    # gps_top_right = [gps_botm_left[0] + lat_ratio * im.shape[0], gps_botm_left[1] + lng_ratio * im.shape[1]]
    #
    p_dic = pickle.load( open( root_folder_path + name_list[iii].replace('/'+str(dialog_phase-1)+'/','/'+ str(dialog_phase) +'/')+".pickle", "rb" ) )
    pos_list = p_dic['pos_list']
    length_of_traj = p_dic['length_of_traj']
    angle_list = p_dic['angle_list']
    action_list = p_dic['action_list']
    attention_list = p_dic['attention_list']
    step_change_of_view_zoom = p_dic['step_change_of_view']
    lng_ratio = p_dic['lng_ratio']
    lat_ratio = p_dic['lat_ratio']
    gps_botm_left = p_dic['gps_botm_left']
    gps_top_right = p_dic['gps_top_right']
    dialog = p_dic['dialog']

    _zoom_speed = 8

    p_dic = pickle.load( open( root_folder_path + name_list[iii].replace('/'+str(dialog_phase)+'/','/0/').replace("_"+str(dialog_phase),'') +".pickle", "rb" ) )
    angle = angle_list[-1]
    destination_index = p_dic['destination_index']
    destination_gps = extracted_xview_landmarks[int(destination_index)][2]

    starting_coord = np.mean(pos_list[0], axis = 0)


    if df['Answer.tag_1'][iii] == df['Answer.tag_1'][iii]:
        answer = df['Answer.tag_0'][iii]+df['Answer.tag_1'][iii]
    else:
        answer = df['Answer.tag_0'][iii]

    dialog += '\n-    Answer: ' + answer
    # print ('\n Previous Dialog: ', complete_instruction)

    # print('-    Question: \n Am I looking at the right semicircle? \nWhere is the destination?')
    print(dialog)
    im_resized = cv2.resize(im_full_map, (int(im_full_map.shape[1] * lng_ratio / lat_ratio), im_full_map.shape[0]), interpolation = cv2.INTER_AREA) # ratio_all = lat_ratio

    #=======================================================
    #    show old traj
    #=======================================================
    __coords = []
    _i = 0
    for i in pos_list:
        _i += 1
        mean_im_coords = np.mean(i, axis=0)
        # print(mean_im_coords)
        if (__coords != []):
            cv2.line(im_resized, (int(mean_im_coords[0]), int(mean_im_coords[1])), __coords[-1], (255, 0, 255), 4)
        __coords.append((int(mean_im_coords[0]), int(mean_im_coords[1])))


    _gps = np.array(
        __coords + [list(starting_coord)])




    # =======================================================
    #    END: old traj shown
    # =======================================================
    im_resized_copy = im_resized.copy()
    corners = pos_list[-1]

    # angle = 0
    width = 1280
    height = 720
    dst_pts = np.array([[0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype="float32")

    mean_im_coords = np.mean(corners, axis=0)

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(corners, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    im_view = cv2.warpPerspective(im_resized, M, (width, height))

    compass_pos = 100
    compass_size = 50
    count_frame = 0

    while True:
        view_ratio = np.linalg.norm(img_to_gps_coords(corners[0]) - img_to_gps_coords(corners[1])) / (max_view[0]/11.13/1e4)\
                     + 0.01
        step_change_of_view = np.array([get_a_gps_coord_at_distance(0, _zoom_speed *6*view_ratio) / lat_ratio,
                                        get_a_gps_coord_at_distance(0, _zoom_speed / width * height *6*view_ratio) / lat_ratio])

        count_frame += 1
        cv2.line(im_view, (compass_pos,compass_pos), (int(compass_pos+20*np.sin(-angle/180*3.14159)), int(compass_pos-20*np.cos(-angle/180*3.14159))),(255,255,255), 2)
        cv2.putText(im_view, 'N', (int(compass_pos+compass_size*np.sin(-angle/180*3.14159)), int(compass_pos-compass_size*np.cos(-angle/180*3.14159))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1, cv2.LINE_AA)

        cv2.line(im_view, (compass_pos, compass_pos), (int(compass_pos + 20 * np.sin((-angle+90) / 180 * 3.14159)), int(compass_pos - 20 * np.cos((-angle+90) / 180 * 3.14159))),
                 (255, 255, 255), 2)
        cv2.putText(im_view, 'E',
                    (int(compass_pos + compass_size * np.sin((-angle+90) / 180 * 3.14159)), int(compass_pos - compass_size * np.cos((-angle+90) / 180 * 3.14159))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(im_view, (compass_pos, compass_pos), (
        int(compass_pos + 20 * np.sin((-angle + 180) / 180 * 3.14159)), int(compass_pos - 20 * np.cos((-angle + 180) / 180 * 3.14159))),
                 (255, 255, 255), 2)
        cv2.putText(im_view, 'S',
                    (int(compass_pos + compass_size * np.sin((-angle + 180) / 180 * 3.14159)),
                     int(compass_pos - compass_size * np.cos((-angle + 180) / 180 * 3.14159))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(im_view, (compass_pos, compass_pos), (
        int(compass_pos + 20 * np.sin((-angle + 270) / 180 * 3.14159)), int(compass_pos - 20 * np.cos((-angle + 270) / 180 * 3.14159))),
                 (255, 255, 255), 2)
        cv2.putText(im_view, 'W',
                    (int(compass_pos + compass_size * np.sin((-angle + 270) / 180 * 3.14159)),
                     int(compass_pos - compass_size * np.cos((-angle + 270) / 180 * 3.14159))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(im_view, (int(width / 4), int(height / 4)), (int(width / 4 + 85), int(height / 4)),
                 (0, 0, 255), 1)
        cv2.line(im_view, (int(width / 4), int(height / 4)), (int(width / 4), int(height / 4 + 85)),
                 (0, 0, 255), 1)

        cv2.line(im_view, (int(3 * width / 4), int(3 * height / 4)), (int(3 * width / 4 - 85), int(3 * height / 4)),
                 (0, 0, 255), 1)
        cv2.line(im_view, (int(3 * width / 4), int(3 * height / 4)), (int(3 * width / 4), int(3 * height / 4 - 85)),
                 (0, 0, 255), 1)

        cv2.line(im_view, (int(3 * width / 4), int(height / 4)), (int(3 * width / 4 - 85), int(height / 4)),
                 (0, 0, 255), 1)
        cv2.line(im_view, (int(3 * width / 4), int(height / 4)), (int(3 * width / 4), int(height / 4 + 85)),
                 (0, 0, 255), 1)

        cv2.line(im_view, (int(width / 4), int(3 * height / 4)), (int(width / 4 + 85), int(3 * height / 4)),
                 (0, 0, 255), 1)
        cv2.line(im_view, (int(width / 4), int(3 * height / 4)), (int(width / 4), int(3 * height / 4 - 85)),
                 (0, 0, 255), 1)


        cv2.line(im_view, (int(width/2-85), int(height/2)), (int(width/2+85), int(height/2)),
                (0, 0, 255), 1)

        cv2.line(im_view, (int(width / 2), int(height / 2 - 85)), (int(width / 2), int(height / 2 + 85)),
                (0, 0, 255), 1)

        cv2.imshow('navigation viewer', im_view)
        k = cv2.waitKey(0)
        action_list.append(k)

        if k == 27:
            approve = ''
            print('\n====== You have pressed ESC for task #', iii, ' =====')

            if starting_pix_dis_to_des < ending_pix_dis_to_des - 100:
                # dialog = 'x ' + dialog
                approve = 'Poor quality, cannot lead to the right direction'
                print('poor quality.')

            print(short_cut)
            your_input = input(
                '\nEnter your question. Or input rej to reject. Or input sentence starting with y to claim the destination.\n')

            for i in range(len(short_cut[0][1:])):
                sc = short_cut.iloc[i, 0]
                substitution_list = [j for j in short_cut.iloc[i, 2:] if j == j]
                print (substitution_list)
                for jj in range(len(substitution_list)):
                    if sc + str(int(jj)) in your_input:
                        substitution = substitution_list[jj]
                        your_input = your_input.replace(sc, substitution)

                if sc in your_input:
                    substitution = random.choice(substitution_list)
                    your_input = your_input.replace(sc, substitution)


            print ('\n[Saved] You just input: \n', your_input)
            if 'xxx' in your_input:
                approve = 'Need further inspection, the phase_0 may be problematic.'
                if pos_list == []:
                    pos_list = [corners]
            elif 'rej' in your_input:
                approve = 'Poor quality, rejected by our manual checking.'
                if pos_list == []:
                    pos_list = [corners]
            else:
                destination_coord = np.array(gps_to_img_coords(destination_gps))
                starting_pix_dis_to_des = np.linalg.norm(
                    np.array(starting_coord) - destination_coord)
                ending_pix_dis_to_des = np.linalg.norm(
                    np.array(np.mean(pos_list[-1], axis=0)) - destination_coord)


                approve = 'X'
                if (len(your_input)>=1 and (your_input[0] == 'Y' or your_input[0] == 'y')) or (len(your_input)>=2 and (your_input[1] == 'Y' or your_input[1] == 'y')):
                    diag_view_area = 0.5 * np.linalg.norm(np.array(pos_list[-1][0]) - np.array(pos_list[-1][2]))
                    diag_destination_coord = np.linalg.norm(np.array(
                        gps_to_img_coords(extracted_xview_landmarks[int(destination_index)][1][0])) - np.array(
                        gps_to_img_coords(extracted_xview_landmarks[int(destination_index)][1][2])))

                    destination_cnt = cv2.UMat(np.array(
                        [gps_to_img_coords(x) for x in extracted_xview_landmarks[int(destination_index)][1]],
                        dtype=np.int32))
                    dist = cv2.pointPolygonTest(destination_cnt, np.mean(pos_list[-1], axis=0), True)


                    # print('Debug: ', (np.abs(diag_view_area/diag_destination_coord - 1), (ending_pix_dis_to_des,min_destination_edge_coord)))
                    if (diag_view_area / diag_destination_coord<1.8 and diag_destination_coord/ diag_view_area<1.8) and dist >= 0:
                        dialog += '\n-    Question: ' + your_input + "\n-    Answer: Yes you have find it!!!"
                        print ("Yes you have find it!!!\n")
                    elif dist >= 0:
                        print(diag_view_area, diag_destination_coord)
                        if diag_view_area > diag_destination_coord:
                            dialog += '\n-    Question: ' + your_input + "\n-    Answer: You need to fly lower."
                            print ("\nAutomatic Answer: You need to fly lower.\n")
                        else:
                            dialog += '\n-    Question: ' + your_input + "\n-    Answer: You need to fly higher."
                            print ("\nAutomatic Answer: You need to fly higher..\n")
                        continue
                    else:
                        dialog += '\n-    Question: ' + your_input + "\n-    Answer: Nope, you haven't get there. Ask some more questions."
                        print ("\nAutomatic Answer: Nope, you haven't get there. Ask some more questions.\n")
                        your_input = input('Enter your new question:\n')

                        for i in range(len(short_cut[0][1:])):
                            sc = short_cut.iloc[i, 0]
                            substitution_list = [j for j in short_cut.iloc[i, 2:] if j == j]
                            print (substitution_list)
                            for jj in range(len(substitution_list)):
                                if sc + str(int(jj)) in your_input:
                                    substitution = substitution_list[jj]
                                    your_input = your_input.replace(sc, substitution)

                            if sc in your_input:
                                substitution = random.choice(substitution_list)
                                your_input = your_input.replace(sc, substitution)

                        print ('\n[Saved] You just input: \n', your_input)

                        dialog += '\n-    Question: ' + your_input


                elif len(your_input) > 0:

                    dialog += '\n-    Question: ' + your_input


                else:
                    assert False

            pickle.dump({'action_list': action_list,
                         'pos_list': pos_list,
                         'length_of_traj': len(pos_list),
                         'angle_list': angle_list,
                         'attention_list':attention_list,
                         'dialog': dialog,
                         'question': your_input,
                         'Approve': approve,
                         'step_change_of_view':step_change_of_view,
                         'lat_ratio':lat_ratio,
                         'lng_ratio':lng_ratio,
                         'gps_botm_left':gps_botm_left,
                         'gps_top_right':gps_top_right
                         }, open(root_folder_path + name_list[iii].replace('/'+str(dialog_phase)+'/', '/'+str(dialog_phase+1)+'/').replace("_"+str(dialog_phase),"_"+str(dialog_phase+1))+'.pickle', 'wb'))
            print ('pickle saved: ',root_folder_path + name_list[iii].replace('/'+str(dialog_phase)+'/', '/'+str(dialog_phase+1)+'/').replace("_"+str(dialog_phase),"_"+str(dialog_phase+1))+'.pickle')









            break
        elif k == ord('2'):

            _new_corners = change_corner(
                corners,
                [step_change_of_view_zoom * np.array([-1, -1]), step_change_of_view_zoom * np.array([1, -1]),
                 step_change_of_view_zoom * np.array([1, 1]), step_change_of_view_zoom * np.array([-1, 1])]
            )
            if np.linalg.norm(img_to_gps_coords(_new_corners[0]) - img_to_gps_coords(_new_corners[1])) >  max_view[0]/11.13/1e4:
                continue

            new_corners = []
            for i in _new_corners:

                if i[0]>0 and i[0] < im_resized.shape[1] and i[1]>0 and i[1]<im_resized.shape[0]:
                    new_corners.append(i)
                else:
                    break
            if len(new_corners) != 4:
                continue
            else:
                corners = np.array(new_corners, dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                im_view = cv2.warpPerspective(im_resized, M, (width, height))

        elif k == ord('1'):

            _new_corners = change_corner(
                corners,
                [step_change_of_view_zoom * np.array([1, 1]), step_change_of_view_zoom * np.array([-1, 1]),
                 step_change_of_view_zoom * np.array([-1, -1]), step_change_of_view_zoom * np.array([1, -1])]
            )

            if np.linalg.norm(img_to_gps_coords(_new_corners[0]) - img_to_gps_coords(_new_corners[1])) <  min_view[0]/11.13/1e4:
                continue

            new_corners = []
            for i in _new_corners:

                if i[0]>0 and i[0] < im_resized.shape[1] and i[1]>0 and i[1]<im_resized.shape[0]:
                    new_corners.append(i)
                else:
                    break
            if len(new_corners) != 4:
                continue
            else:
                corners = np.array(new_corners, dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                im_view = cv2.warpPerspective(im_resized, M, (width, height))

        elif k == ord('w'):

            _new_corners = change_corner(
                corners,
                [step_change_of_view * np.array([0, -1]), step_change_of_view * np.array([0, -1]),
                 step_change_of_view * np.array([0, -1]), step_change_of_view * np.array([0, -1])]
            )


            new_corners = []
            for i in _new_corners:

                if i[0]>0 and i[0] < im_resized.shape[1] and i[1]>0 and i[1]<im_resized.shape[0]:
                    new_corners.append(i)
                else:
                    break
            if len(new_corners) != 4:
                continue
            else:
                corners = np.array(new_corners, dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                im_view = cv2.warpPerspective(im_resized, M, (width, height))
        elif k == ord('s'):


            _new_corners = change_corner(
                corners,
                [step_change_of_view * np.array([0, 1]), step_change_of_view * np.array([0, 1]),
                 step_change_of_view * np.array([0, 1]), step_change_of_view * np.array([0, 1])]
            )
            new_corners = []
            for i in _new_corners:

                if i[0]>0 and i[0] < im_resized.shape[1] and i[1]>0 and i[1]<im_resized.shape[0]:
                    new_corners.append(i)
                else:
                    break
            if len(new_corners) != 4:
                continue
            else:
                corners = np.array(new_corners, dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                im_view = cv2.warpPerspective(im_resized, M, (width, height))
        elif k == ord('d'):

            _new_corners = change_corner(
                corners,
                [step_change_of_view * np.array([1, 0]), step_change_of_view * np.array([1, 0]),
                 step_change_of_view * np.array([1, 0]), step_change_of_view * np.array([1, 0])]
            )
            new_corners = []
            for i in _new_corners:
                if i[0]>0 and i[0] < im_resized.shape[1] and i[1]>0 and i[1]<im_resized.shape[0]:
                    new_corners.append(i)
                else:
                    break
            if len(new_corners) != 4:
                continue
            else:
                corners = np.array(new_corners, dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                im_view = cv2.warpPerspective(im_resized, M, (width, height))

        elif k == ord('a'):

            _new_corners = change_corner(
                corners,
                [step_change_of_view * np.array([-1, 0]), step_change_of_view * np.array([-1, 0]),
                 step_change_of_view * np.array([-1, 0]), step_change_of_view * np.array([-1, 0])]
            )
            new_corners = []
            for i in _new_corners:

                if i[0]>0 and i[0] < im_resized.shape[1] and i[1]>0 and i[1]<im_resized.shape[0]:
                    new_corners.append(i)
                else:
                    break
            if len(new_corners) != 4:
                continue
            else:
                corners = np.array(new_corners, dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(corners, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                im_view = cv2.warpPerspective(im_resized, M, (width, height))

        elif k == ord('e'):
            angle += 10

            mean_im_coords = np.mean(corners,axis = 0)
            _corners = [
                corners[0] - mean_im_coords,
                corners[1] - mean_im_coords,
                corners[2] - mean_im_coords,
                corners[3] - mean_im_coords
            ]# counter clock wise
            rotated_corners = []
            for i in range(4):
                rotated_point = mean_im_coords + rotation_anticlock(10, _corners[i])
                if rotated_point[0] > 0 and rotated_point[0] < im_resized.shape[1] and rotated_point[1] > 0 and rotated_point[1] < im_resized.shape[0]:
                    rotated_corners.append(rotated_point)
                else:
                    break
            if len(rotated_corners) != 4:
                angle-=10
                continue
            corners = np.array(rotated_corners, dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(corners, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            im_view = cv2.warpPerspective(im_resized, M, (width, height))

        elif k == ord('q'):
            angle -= 10

            mean_im_coords = np.mean(corners, axis=0)
            _corners = [
                corners[0] - mean_im_coords,
                corners[1] - mean_im_coords,
                corners[2] - mean_im_coords,
                corners[3] - mean_im_coords
            ]  # counter clock wise
            rotated_corners = []
            for i in range(4):
                rotated_point = mean_im_coords + rotation_anticlock(-10, _corners[i])
                if rotated_point[0] > 0 and rotated_point[0] < im_resized.shape[1] and rotated_point[1] > 0 and rotated_point[1] < im_resized.shape[0]:
                    rotated_corners.append(rotated_point)
                else:
                    break
            if len(rotated_corners) != 4:
                angle+=10
                continue
            corners = np.array(rotated_corners, dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(corners, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            im_view = cv2.warpPerspective(im_resized, M, (width, height))


        angle_list.append(angle)
        pos_list.append(corners)

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
        # cv2.rectangle(im_resized_copy, (int(i[0][0]),int(i[0][1])), (int(i[2][0]),int(i[2][1])), (255, 255, 255), 1)

    # print((np.array(__coords,dtype = np.int32),\
    #     np.array([list(starting_coord)] + [list(gps_to_img_coords((destination_gps)))] + [gps_to_img_coords(i) for i in extracted_xview_landmarks[destination_index][1]])
    #                   ))

    _gps = np.vstack((np.vstack(__coords), \
                      np.array(
                          [list(starting_coord)] + [list(gps_to_img_coords((destination_gps)))] + [gps_to_img_coords(i)
                                                                                                   for i in
                                                                                                   extracted_xview_landmarks[
                                                                                                       destination_index][
                                                                                                       1]])
                      ))
    im_min_boundary = _gps.min(axis=0)
    im_max_boundary = _gps.max(axis=0)

    # print(_gps)
    # print(im_min_boundary, im_max_boundary)

    size_boundary = np.array(im_max_boundary) - np.array(im_min_boundary) + np.array([100, 100])
    center_coord = np.mean(pos_list[-1], axis = 0)

    im_min_boundary[0] = int(max(center_coord[0] - size_boundary[0], 0))
    im_min_boundary[1] = int(max(center_coord[1] - size_boundary[1], 0))

    im_max_boundary[0] = int(min(center_coord[0] + size_boundary[0], im_resized_copy.shape[1]))
    im_max_boundary[1] = int(min(center_coord[1] + size_boundary[1], im_resized_copy.shape[0]))

    compass_size = int( np.linalg.norm(np.array(pos_list[-1][0]) - np.array(pos_list[-1][2])) / 20) + 80
    compass_size_center = int(compass_size * 0.45)

    compass_size_edge = int(compass_size * 0.75)

    cv2.line(im_resized_copy,
             (
                 int(center_coord[0] + compass_size_edge * 0.35 * np.sin((angle + 135) / 180 * 3.14159)),
                 int(center_coord[1] - compass_size_edge * 0.35 * np.cos((angle + 135) / 180 * 3.14159))
             ),
             (int(center_coord[0]), int(center_coord[1])),
             (0, 0, 255), 2 + int(size_boundary[0] / 400))

    cv2.line(im_resized_copy,
             (
                 int(center_coord[0] + compass_size_edge * 0.35 * np.sin((angle + 225) / 180 * 3.14159)),
                 int(center_coord[1] - compass_size_edge * 0.35 * np.cos((angle + 225) / 180 * 3.14159))
             ),
             (int(center_coord[0]), int(center_coord[1])),
             (0, 0, 255), 2 + int(size_boundary[0] / 400))

    cv2.line(im_resized_copy,
             (
                 int(center_coord[0] + compass_size_edge*0.7 * np.sin((angle + 180) / 180 * 3.14159)),
                 int(center_coord[1] - compass_size_edge*0.7 * np.cos((angle + 180) / 180 * 3.14159))
             ),
             (int(center_coord[0]), int(center_coord[1])),
             (0, 0, 255), 2 + int(size_boundary[0] / 400))
    # for k in range(12):
    #
    #     if k == 0 or k ==3 or k==6 or k ==9:
    #         cv2.line(im_resized_copy,
    #                  (
    #                      int(center_coord[0] + compass_size_edge * 1.4 * np.sin((angle + k * 30) / 180 * 3.14159)),
    #                      int(center_coord[1] - compass_size_edge * 1.4 * np.cos((angle + k * 30) / 180 * 3.14159))
    #                  ),
    #                  (
    #                      int(center_coord[0] + compass_size_center * np.sin((angle + k * 30) / 180 * 3.14159)),
    #                      int(center_coord[1] - compass_size_center * np.cos((angle + k * 30) / 180 * 3.14159))
    #                  ),
    #                  (255, 255, 255), 1 + int(size_boundary[0] / 800))
    #
    #     else:
    #         cv2.line(im_resized_copy,
    #                  (
    #                      int(center_coord[0] + compass_size_edge * np.sin((angle + k * 30) / 180 * 3.14159)),
    #                      int(center_coord[1] - compass_size_edge * np.cos((angle + k * 30) / 180 * 3.14159))
    #                  ),
    #                  (
    #                      int(center_coord[0] + compass_size_center * np.sin((angle + k * 30) / 180 * 3.14159)),
    #                      int(center_coord[1] - compass_size_center * np.cos((angle + k * 30) / 180 * 3.14159))
    #                  ),
    #                  (255, 255, 255), 1 + int(size_boundary[0] / 800))

    cv2.putText(im_resized_copy, 'Forward direction', (int(center_coord[0]) ,int(center_coord[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.1 + size_boundary[0] / 1200, (0, 0, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)
    #
    # cv2.putText(im_resized_copy, '3',
    #             (int(center_coord[0] + compass_size * np.sin((angle + 90) / 180 * 3.14159)),
    #              int(center_coord[1] - compass_size * np.cos((angle + 90) / 180 * 3.14159))),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5 + size_boundary[0] / 1200, (255, 255, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)
    #
    # cv2.putText(im_resized_copy, '6',
    #             (int(center_coord[0] + compass_size * np.sin((angle + 180) / 180 * 3.14159)),
    #              int(center_coord[1] - compass_size * np.cos((angle + 180) / 180 * 3.14159))),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5 + size_boundary[0] / 1200, (255, 255, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)
    #
    # cv2.putText(im_resized_copy, '9',
    #             (int(center_coord[0] + compass_size * np.sin((angle + 270) / 180 * 3.14159)),
    #              int(center_coord[1] - compass_size * np.cos((angle + 270) / 180 * 3.14159))),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5 + size_boundary[0] / 1200, (255, 255, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)

    cv2.putText(im_resized_copy, 'current view area', np.array(np.min(pos_list[-1],axis = 0),dtype=np.int32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 + size_boundary[0] / 1600, (255, 255, 255), 1 + int(size_boundary[0] / 500), cv2.LINE_AA)


    # print(im_min_boundary[1],im_max_boundary[1], im_min_boundary[0],im_max_boundary[0])

    cv2.rectangle(im_resized_copy, gps_to_img_coords(extracted_xview_landmarks[int(destination_index)][1][0]),
                  gps_to_img_coords(extracted_xview_landmarks[int(destination_index)][1][2]), (255, 0, 255), 2)

    cv2.putText(im_resized_copy, 'Destination', np.array(gps_to_img_coords(destination_gps)) + np.array(
        [int(polygon_area(extracted_xview_landmarks[int(destination_index)][1]) * 30000),
         int(polygon_area(extracted_xview_landmarks[int(destination_index)][1]) * 30000)]), \
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 + size_boundary[0] / 800, (255, 255, 255), 1 + int(size_boundary[0] / 300), cv2.LINE_AA)
    print ('[Debug]: ' + root_folder_path + name_list[iii].replace('/'+str(dialog_phase)+'/', '/'+str(dialog_phase+1)+'/image_sample_')[:-1] + str(dialog_phase+1) + '.jpg')
    cv2.imwrite(root_folder_path + name_list[iii].replace('/'+str(dialog_phase)+'/', '/'+str(dialog_phase+1)+'/image_sample_')[:-1] + str(dialog_phase+1) + '.jpg', im_resized_copy[int(im_min_boundary[1]):int(im_max_boundary[1]), int(im_min_boundary[0]):int(im_max_boundary[0])])


    # gt = cv2.imread(root_folder_path + name_list[iii].replace('/', '/0/image_sample_') + '.jpg', 1)
    # cv2.imshow('GT routes', gt)
    # cv2.waitKey(0)
cv2.destroyAllWindows()
# set the push button feed back record the actions

# set a end button and compare/save the route