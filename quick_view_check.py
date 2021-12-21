import os

import json
import cv2
import numpy as np
import pickle
import pandas as pd
import scipy.spatial as spt
import  random


root_folder_path = '/Users/fanyue/xview/'
df = pd.read_csv('/Users/fanyue/Downloads/Batch_4632300_batch_results.csv_filtered.csv')


# open a opencv window and display the initial view
cv2.namedWindow('navigation viewer')

for iii in range(0 ,len(df['Input.task_image_name'])):
    if 'clock' in df['Answer.tag'][iii] and (df.loc[iii, 'Reject']) != (df.loc[iii, 'Reject']):
        path = root_folder_path + df['Input.task_image_name'][iii]
        print(path)
        im_full_map = cv2.imread(path, 1)
        cv2.imshow('viewer', im_full_map)
        print(df['Answer.tag'][iii])
        k = cv2.waitKey(0)
        if k == ord('x'):
            df.loc[iii, 'Reject'] = 'Wrong use of clock direction. The clock direction should be from the drone\'s perspective, so the 12 o\'clock direction is the drone\'s  forward direction. '

        df.to_csv('/Users/fanyue/Downloads/Batch_4632300_batch_results.csv_second_filtered.csv', index=False)