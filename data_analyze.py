import argparse
import numpy as np
import cv2 as cv2
import os
import os.path as op
import glob

from utils import *

dir_x = '/home/hu440/LP-MOT/Dataset/2021lpcvc_test/flows/5p4b_competition/labels'
dir_y = '/home/hu440/LP-MOT/Dataset/2021lpcvc_test/flows/5p4b_competition/opencv'

aee_list = []
for i, flo_x in enumerate(os.listdir(dir_x)):
    flo_x = op.join(dir_x, flo_x)
    flo_y = op.join(dir_y, sorted(os.listdir(dir_y))[i])

    print(f' - label size: {read_flo_file(flo_x).shape}')
    print(f' - opencv size: {read_flo_file(flo_y).shape}')

    aee = evaluate_flow_file(flo_x, flo_y)
    aee_list.append(aee)
    print(f' - AEE score: {aee}')

print(f' ==== AEE score: {sum(aee_list)/len(aee_list)} ====')
