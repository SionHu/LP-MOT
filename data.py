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
required.add_argument('-o', '--output_path', metavar="OUTPUT",
                      help='Output path of the optical flow video')
optional.add_argument('-f', '--frames', action='store_true',
                    help='Input is a squence of frames instead of one video')
optional.add_argument('-of', '--flo', action='store_true',
                    help='Generate .flo files from the rgb images')
optional.add_argument('-s', '--stopper', type=int,
                    help='Used to stop earlier for testing')
parser._action_groups.append(optional)
args = parser.parse_args()

def optical_flow_from_video():
    # Get a VideoCapture object from video and store it in vs
    vc = cv2.VideoCapture(args.input_path)
    # Read first frame
    ret, first_frame = vc.read()

    # Scale and resize image
    resize_dim = 640
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    # Convert to gray scale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Create mask
    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{args.output_path}/output.mp4',fourcc,10,(600, 600))

    count = 0
    err_count = 0
    while(vc.isOpened()):
        print(f' - reading frame {count}')

        # Stop early for testing
        if args.stopper and count == args.stopper:
            break

        # Read a frame from video
        ret, frame = vc.read()
        # Convert new frame format`s to gray scale and resize gray frame obtained
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        except Exception as e:
            print(' - There is a problem!')
            err_count += 1
            continue

        # Calculate dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)

        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Resize frame size to match dimensions
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Open a new window and displays the output frame
        dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)

        # Save the output files (rgb or flo)
        if args.flo:
            filename = f'{args.output_path}/{count}.flo'
            write_flo(flow, filename)
        else:
            cv2.imwrite(f'{args.output_path}/{count}.jpg', dense_flow)
            # cv2.imshow("Dense optical flow", dense_flow)s
            out.write(dense_flow)

        # Update previous frame
        prev_gray = gray
        # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        count += 1

    print(f' ==== err_count: {err_count} ====')

    # The following frees up resources and closes all windows
    vc.release()
    cv2.destroyAllWindows()

def optical_flow_from_frames():

    # Scale and resize image
    first_frame_path = op.join(args.input_path, sorted(os.listdir(args.input_path))[0])
    first_frame = cv2.imread(first_frame_path)
    resize_dim = 600
    max_dim = max(first_frame.shape)
    scale = resize_dim/max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    # Convert to gray scale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Create mask
    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{args.output_path}/output.mp4',fourcc,10,(600, 600))

    # Input is a squence of frames
    for i,filename in enumerate(sorted(os.listdir(args.input_path))):
        print(f' -- reading {filename} ...')
        # Stop early for testing
        if args.stopper and i == args.stopper:
            break
        # Current image
        frame_path = op.join(args.input_path, filename)
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Calculate dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Resize frame size to match dimensions
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Open a new window and displays the output frame
        dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)

        cv2.imwrite(f'{args.output_path}/{filename}', dense_flow)
        out.write(dense_flow)
        # cv2.imshow("Dense optical flow", dense_flow)
        # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Update previous frame
        prev_gray = gray

    cv2.destroyAllWindows()

def main():
    if args.frames:
        optical_flow_from_frames()
    else:
        optical_flow_from_video()


if __name__ == '__main__':
    main()
