import csv
import argparse
import os
import cv2
from datetime import datetime

def distEstimate(H, h):
    """
        H: height of image (pixel)
        h: lower ordinate from top of the image (pixel)
    """
    aspect_ratio: float = 4/3
    fov: float = math.radians(84)

    Hc: float = 10.60000034
    gimbal_pitch: float = math.radians(3.1)
    drone_pitch: float = math.radians(51.7)
    roll: float = math.radians(0)

    pitch = gimbal_pitch+drone_pitch
    lens_scaling = math.hypot(1, aspect_ratio) / math.tan(fov / 2)
    horizon = (H / 2) * (1 - math.tan(pitch) * lens_scaling)
    # horizon = (H / 2) * (1 - math.sin(pitch/2)*2 * lens_scaling)
    scaled_height = lens_scaling * (H / 2) / ((h - horizon) * math.cos(roll))
    distance = Hc * scaled_height / (math.cos(pitch) ** 2) - Hc * math.tan(pitch)
    print('\ndistance: ', distance)

    return distance

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--flight_log', required=True, help='Flight log in csv\n')
    parser.add_argument('-v', '--video', required=True, help='Video corresponding to the flight log\n')
    args = parser.parse_args()

    # Read video input: e.g. 2021-03-13-10-57-32-431.mp4
    sMinSec = "-".join(args.video[:-4].split('-')[4:6])
    sMillsec = " ".join(args.video[:-4].split('-')[6:8])+'00'
    k = int(int(sMillsec)/10000)

    # Read and store flight logs: e.g. Mar-13th-2021-10-57AM-Flight-Airdata.csv
    flogs = {} # key is millisecond, value is OrderedDict([(k, v), ..., (k, v)])
    with open(args.flight_log, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        i, j, boo = 0, 0, False
        for row in reader:
            # key is stored in format min-sec-millisecond e.g. 58-41-88600
            milliSec = row['time(millisecond)']
            minSec =  row['datetime(utc)'].split(':')[1:3]
            minSecMill = f'{minSec[0]}-{minSec[1]}-{milliSec}'
            if i >= k-1 and boo:
                flogs[minSecMill] = row
            if sMinSec in minSecMill:
                boo = True
                i += 1
            elif boo:
                flogs[minSecMill] = row

        print(flogs['57-32-19200'])


    # Find the starting row in the flogs
    # sKeyList = [key for key, value in flogs.items() if sMinSec in key]
    # sKey = sKeyList[k-1]

    # cap = cv2.VideoCapture(args.video)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    #
    # timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    # calc_timestamps = [0.0]
    # frameCount = []
    # i = 0
    # while(cap.isOpened()):
    #     frame_exists, curr_frame = cap.read()
    #     if frame_exists:
    #         timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    #         calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    #         frameCount.append(i)
    #         i += 1
    #     else:
    #         break
    #
    # cap.release()
    #
    # for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    #     fMillisec = int(i * 1000 / fps + int(sMillsec)/100)


if __name__ == '__main__':
    main()
