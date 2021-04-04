import csv
import argparse
import os
import cv2


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--flight_log', required=True, help='Flight log in csv\n')
    parser.add_argument('-v', '--video', required=True, help='Video corresponding to the flight log\n')
    args = parser.parse_args()

    # Read and store flight logs: e.g. Mar-13th-2021-10-57AM-Flight-Airdata.csv
    flogs = {} # key is millisecond, value is OrderedDict([(k, v), ..., (k, v)])
    with open(args.flight_log, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # TODO: make key in time format e.g. 2021-03-13 15:58:44
            flogs[row['\ufefftime(millisecond)']] = row
    print(flogs['91300'])

    # Read video input: e.g. 2021-03-13-10-57-32-431.mp4
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]

    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        else:
            break

    cap.release()

    for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
        # print('Frame %d difference:'%i, abs(ts - cts))
        print('Frame %d difference:'%i, ts)



if __name__ == '__main__':
    main()
