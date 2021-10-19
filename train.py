import argparse
import glob
import numpy as np
import cv2 as cv2
import os
import os.path as op
import glob

from utils import *

# ==== Command Line Arguments ====
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input_path', metavar="INPUT", required=True,
                      help='Input path to the video frames')
optional.add_argument('-s', '--stopper', type=int,
                    help='Used to stop earlier for testing')
parser._action_groups.append(optional)
args = parser.parse_args()
