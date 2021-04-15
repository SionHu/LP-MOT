from enum import Enum
import logging
import time
import cv2
import numpy as np
import numba as nb
from math import asin,cos,pi,sin

from .detector import SSDDetector, YoloDetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow


LOGGER = logging.getLogger(__name__)

import fractions, math
@nb.jit(fastmath=True, cache=True)
def distEstimate(H, h, Hc, gPitch, dPitch):
    """
        H: height of image (pixel)
        h: lower ordinate from top of the image (pixel)
    """
    aspect_ratio: float = 4/3
    fov: float = math.radians(84)

    # Hc: float = 10.60000034
    gimbal_pitch: float = math.radians(-gPitch)
    drone_pitch: float = math.radians(dPitch)
    roll: float = math.radians(0)

    pitch = gimbal_pitch+drone_pitch
    lens_scaling = math.hypot(1, aspect_ratio) / math.tan(fov / 2)
    horizon = (H / 2) * (1 - math.tan(pitch) * lens_scaling)
    # horizon = (H / 2) * (1 - math.sin(pitch/2)*2 * lens_scaling)
    scaled_height = lens_scaling * (H / 2) / ((h - horizon) * math.cos(roll))
    distance = Hc * scaled_height / (math.cos(pitch) ** 2) - Hc * math.tan(pitch)
    # print('\ndistance: ', distance)

    return distance

class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    """
    This is the top level module that integrates detection, feature extraction,
    and tracking together.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    capture_dt : float
        Time interval in seconds between each captured frame.
    config : Dict
        Tracker configuration.
    draw : bool
        Flag to toggle visualization drawing.
    verbose : bool
        Flag to toggle output verbosity.
    """

    def __init__(self, size, capture_dt, config, flogs, fps, sMillsec, draw=False, verbose=False):
        self.size = size
        self.draw = draw
        self.verbose = verbose
        self.detector_type = DetectorType[config['detector_type']]
        self.detector_frame_skip = config['detector_frame_skip']
        self.flogs = flogs
        self.fps = fps
        self.sMillsec = sMillsec

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, config['ssd_detector'])
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YoloDetector(self.size, config['yolo_detector'])
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, config['public_detector'])

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(config['feature_extractor'])
        self.tracker = MultiTracker(self.size, capture_dt, self.extractor.metric,
                                    config['multi_tracker'])

        # reset counters
        self.frame_count = 0
        self.detector_frame_count = 0
        self.preproc_time = 0
        self.detector_time = 0
        self.extractor_time = 0
        self.association_time = 0
        self.tracker_time = 0

    @property
    def visible_tracks(self):
        # retrieve confirmed and active tracks from the tracker
        return [track for track in self.tracker.tracks.values()
                if track.confirmed and track.active]

    def initiate(self):
        """
        Resets multiple object tracker.
        """
        self.frame_count = 0

    def step(self, frame):
        """
        Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        detections = []
        if self.frame_count == 0:
            detections = self.detector(self.frame_count, frame)
            self.calcPosition(detections) # update detections to track
            self.tracker.initiate(frame, detections)
        else:
            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                self.detector.detect_async(self.frame_count, frame)
                self.preproc_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.tracker.compute_flow(frame)
                detections = self.detector.postprocess()
                self.calcPosition(detections) # update detections to track
                self.detector_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.extractor.extract_async(frame, detections)
                self.tracker.apply_kalman()
                embeddings = self.extractor.postprocess()
                self.extractor_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.tracker.update(self.frame_count, detections, embeddings)
                self.association_time += time.perf_counter() - tic
                self.detector_frame_count += 1
            else:
                tic = time.perf_counter()
                self.tracker.track(frame)
                self.tracker_time += time.perf_counter() - tic

        if self.draw:
            self._draw(frame, detections)
        self.frame_count += 1

    def _draw(self, frame, detections):
        draw_tracks(frame, self.visible_tracks, show_flow=self.verbose)
        if self.verbose:
            draw_detections(frame, detections)
            draw_flow_bboxes(frame, self.tracker)
            draw_background_flow(frame, self.tracker)
        cv2.putText(frame, f'visible: {len(self.visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)

    def calcPosition(self, detections):
        # TODO: Maybe speeding this part up by turnning dict into list
        list_i = int(self.frame_count * 10 / self.fps)
        dict_i = list(self.flogs)[list_i]

        for det in detections:
            if det.label == 1:
                H, h = self.size[1], det.tlbr[2]
                # H, h = 2160, det.tlbr[2]
                row = self.flogs[dict_i]
                Hc, gPitch, dPitch = (float(row['height_above_takeoff(feet)'])*0.3048,
                                        float(row['gimbal_pitch(degrees)']), float(row['pitch(degrees)']))
                dist = distEstimate(H, h, Hc, gPitch, dPitch)
                lat1, lon1, bearing = (float(row['latitude']), float(row['longitude']),
                                            float(row['compass_heading(degrees)']))
                (lat, lon) = pointRadialDistance(lat1, lon1, bearing, dist/1000)
                det.dist = dist
                det.gps = (lat, lon)
                self.tracker.dist = dist
                self.tracker.dist = (lat, lon)

                print("H \t\t h \t\t Hc \t gPitch \t\t dPitch \t\t dist \t\t tlbr")
                print("%f \t %f \t %f \t %f \t %f \t %f \t %s" % (H, h, Hc, gPitch, dPitch, dist, det.tlbr))


rEarth = 6371.01 # Earth's average radius in km
epsilon = 0.000001 # threshold for floating-point equality

def deg2rad(angle):
    return angle*pi/180

def rad2deg(angle):
    return angle*180/pi

def pointRadialDistance(lat1, lon1, bearing, distance):
    """
    Return final coordinates (lat2,lon2) [in degrees] given initial coordinates
    (lat1,lon1) [in degrees] and a bearing [in degrees] and distance [in km]
    """
    rlat1 = deg2rad(lat1)
    rlon1 = deg2rad(lon1)
    rbearing = deg2rad(bearing)
    rdistance = distance / rEarth # normalize linear distance to radian angle

    rlat = asin( sin(rlat1) * cos(rdistance) + cos(rlat1) * sin(rdistance) * cos(rbearing) )

    if cos(rlat) == 0 or abs(cos(rlat)) < epsilon: # Endpoint a pole
        rlon=rlon1
    else:
        rlon = ( (rlon1 - asin( sin(rbearing)* sin(rdistance) / cos(rlat) ) + pi ) % (2*pi) ) - pi

    lat = rad2deg(rlat)
    lon = rad2deg(rlon)
    return (lat, lon)
