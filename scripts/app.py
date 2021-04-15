#!/usr/bin/env python3

from pathlib import Path
import argparse
import logging
import time
import json
import cv2

import fastmot


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    parser.add_argument('-i', '--input_uri', metavar="URI", required=True, help=
                        'URI to input stream\n'
                        '1) video file (e.g. input.mp4)\n'
                        '2) MIPI CSI camera (e.g. csi://0)\n'
                        '3) USB or V4L2 camera (e.g. /dev/video0)\n'
                        '4) RTSP stream (rtsp://<user>:<password>@<ip>:<port>)\n')
    parser.add_argument('-o', '--output_uri', metavar="URI",
                        help='URI to output stream (e.g. output.mp4)')
    parser.add_argument('-l', '--log', metavar="FILE",
                        help='output a MOT Challenge format log (e.g. eval/results/mot17-04.txt)')
    parser.add_argument('-g', '--gui', action='store_true', help='enable display')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output for debugging')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # load config file
    with open(Path(__file__).parent / 'cfg' / 'mot.json') as config_file:
        config = json.load(config_file, cls=fastmot.utils.ConfigDecoder)

    mot = None
    log = None
    elapsed_time = 0
    stream = fastmot.VideoIO(config['size'], config['video_io'], args.input_uri, args.output_uri)

    if args.mot:
        draw = args.gui or args.output_uri is not None
        mot = fastmot.MOT(config['size'], stream.capture_dt, config['mot'],
                          draw=draw, verbose=args.verbose)
        if args.log is not None:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.log, 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        tic = time.perf_counter()
        while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
            frame = stream.read()
            if frame is None:
                break

            if args.mot:
                mot.step(frame)
                if log is not None:
                    for track in mot.visible_tracks:
                        # MOT17 dataset is usually of size 1920x1080, modify this otherwise
                        orig_size = (1920, 1080)
                        tl = track.tlbr[:2] / config['size'] * orig_size
                        br = track.tlbr[2:] / config['size'] * orig_size
                        w, h = br - tl + 1
                        log.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                  f'{w:.6f},{h:.6f},-1,-1,-1\n')

            if args.gui:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if args.output_uri is not None:
                stream.write(frame)

        toc = time.perf_counter()
        elapsed_time = toc - tic
    finally:
        # clean up resources
        if log is not None:
            log.close()
        stream.release()
        cv2.destroyAllWindows()

    if args.mot:
        # timing results
        avg_fps = round(mot.frame_count / elapsed_time)
        avg_tracker_time = mot.tracker_time / (mot.frame_count - mot.detector_frame_count)
        avg_extractor_time = mot.extractor_time / mot.detector_frame_count
        avg_preproc_time = mot.preproc_time / mot.detector_frame_count
        avg_detector_time = mot.detector_time / mot.detector_frame_count
        avg_assoc_time = mot.association_time / mot.detector_frame_count

        logger.info('Average FPS: %d', avg_fps)
        logger.debug('Average tracker time: %f', avg_tracker_time)
        logger.debug('Average feature extractor time: %f', avg_extractor_time)
        logger.debug('Average preprocessing time: %f', avg_preproc_time)
        logger.debug('Average detector time: %f', avg_detector_time)
        logger.debug('Average association time: %f', avg_assoc_time)


if __name__ == '__main__':
    main()
