from pathlib import Path
from enum import Enum
from collections import deque
import subprocess
import threading
import logging
import cv2


LOGGER = logging.getLogger(__name__)
WITH_GSTREAMER = True


class Protocol(Enum):
    FILE = 0
    CSI = 1
    V4L2 = 2
    RTSP = 3


class VideoIO:
    """
    Class for video capturing from video files or cameras, and writing video files.
    Encoding, decoding, and scaling can be accelerated using the GStreamer backend.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame to output.
    config : Dict
        Camera and buffer configuration.
    input_uri : string
        URI to an input video file or capturing device.
    output_uri : string
        URI to an output video file.
    proc_fps : int
        Estimated processing speed. This depends on compute and scene complexity.
    """

    def __init__(self, size, config, input_uri, output_uri=None, proc_fps=30):
        self.size = size
        self.input_uri = input_uri
        self.output_uri = output_uri

        self.camera_size = config['camera_size']
        self.camera_fps = config['camera_fps']
        self.buffer_size = config['buffer_size']

        self.protocol = self._parse_uri(self.input_uri)
        if WITH_GSTREAMER:
            self.cap = cv2.VideoCapture(self._gst_cap_pipeline(), cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.input_uri)

        self.frame_queue = deque([], maxlen=self.buffer_size)
        self.cond = threading.Condition()
        self.exit_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_frames)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError('Unable to read video stream')
        self.frame_queue.append(frame)

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.do_resize = (width, height) != self.size
        if self.fps == 0:
            self.fps = self.camera_fps # fallback
        LOGGER.info('%dx%d stream @ %d FPS', width, height, self.fps)

        if self.protocol == Protocol.FILE:
            self.capture_dt = 1 / self.fps
        else:
            # limit capture interval at processing latency for camera
            self.capture_dt = 1 / min(self.fps, proc_fps)

        if self.output_uri is not None:
            Path(self.output_uri).parent.mkdir(parents=True, exist_ok=True)
            output_fps = 1 / self.capture_dt
            if WITH_GSTREAMER:
                self.writer = cv2.VideoWriter(self._gst_write_pipeline(), cv2.CAP_GSTREAMER, 0,
                                              output_fps, self.size, True)
            else:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                self.writer = cv2.VideoWriter(self.output_uri, fourcc, output_fps, self.size, True)

    def start_capture(self):
        """
        Start capturing from video file or device.
        """
        if not self.cap.isOpened():
            self.cap.open(self._gst_cap_pipeline(), cv2.CAP_GSTREAMER)
        if not self.capture_thread.is_alive():
            self.capture_thread.start()

    def stop_capture(self):
        """
        Stop capturing from video file or device.
        """
        with self.cond:
            self.exit_event.set()
            self.cond.notify()
        self.frame_queue.clear()
        self.capture_thread.join()

    def read(self):
        """
        Returns the next video frame.
        Returns None if there are no more frames.
        """
        with self.cond:
            while len(self.frame_queue) == 0 and not self.exit_event.is_set():
                self.cond.wait()
            if len(self.frame_queue) == 0 and self.exit_event.is_set():
                return None
            frame = self.frame_queue.popleft()
            self.cond.notify()
        if self.do_resize:
            frame = cv2.resize(frame, self.size)
        return frame

    def write(self, frame):
        """
        Writes the next video frame.
        """
        assert hasattr(self, 'writer')
        self.writer.write(frame)

    def release(self):
        """
        Closes video file or capturing device.
        """
        self.stop_capture()
        if hasattr(self, 'writer'):
            self.writer.release()
        self.cap.release()

    def _gst_cap_pipeline(self):
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        if 'nvvidconv' in gst_elements and self.protocol != Protocol.V4L2:
            # format conversion for hardware decoder
            cvt_pipeline = (
                'nvvidconv interpolation-method=5 ! '
                'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx !'
                'videoconvert ! appsink sync=false'
                % self.size
            )
        else:
            cvt_pipeline = (
                'videoscale ! '
                'video/x-raw, width=(int)%d, height=(int)%d !'
                'videoconvert ! appsink sync=false'
                % self.size
            )

        if self.protocol == Protocol.FILE:
            pipeline = 'filesrc location=%s ! decodebin ! ' % self.input_uri
        elif self.protocol == Protocol.CSI:
            if 'nvarguscamerasrc' in gst_elements:
                pipeline = (
                    'nvarguscamerasrc sensor_id=%s ! '
                    'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, '
                    'format=(string)NV12, framerate=(fraction)%d/1 ! '
                    % (
                        self.input_uri[6:],
                        *self.camera_size,
                        self.camera_fps
                    )
                )
            else:
                raise RuntimeError('GStreamer CSI plugin not found')
        elif self.protocol == Protocol.V4L2:
            if 'v4l2src' in gst_elements:
                pipeline = (
                    'v4l2src device=%s ! '
                    'video/x-raw, width=(int)%d, height=(int)%d, '
                    'format=(string)YUY2, framerate=(fraction)%d/1 ! '
                    % (
                        self.input_uri,
                        *self.camera_size,
                        self.camera_fps
                    )
                )
            else:
                raise RuntimeError('GStreamer V4L2 plugin not found')
        elif self.protocol == Protocol.RTSP:
            pipeline = 'rtspsrc location=%s latency=0 ! decodebin ! ' % self.input_uri
        return pipeline + cvt_pipeline

    def _gst_write_pipeline(self):
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        # use hardware encoder if found
        if 'omxh264enc' in gst_elements:
            h264_encoder = 'omxh264enc'
        elif 'x264enc' in gst_elements:
            h264_encoder = 'x264enc'
        else:
            raise RuntimeError('GStreamer H.264 encoder not found')
        pipeline = (
            'appsrc ! autovideoconvert ! %s ! qtmux ! filesink location=%s '
            % (
                h264_encoder,
                self.output_uri
            )
        )
        return pipeline

    def _capture_frames(self):
        while not self.exit_event.is_set():
            ret, frame = self.cap.read()
            with self.cond:
                if not ret:
                    self.exit_event.set()
                    self.cond.notify()
                    break
                # keep unprocessed frames in the buffer for video file
                if self.protocol == Protocol.FILE:
                    while (len(self.frame_queue) == self.buffer_size and
                           not self.exit_event.is_set()):
                        self.cond.wait()
                self.frame_queue.append(frame)
                self.cond.notify()

    @staticmethod
    def _parse_uri(uri):
        pos = uri.find('://')
        if '/dev/video' in uri:
            protocol = Protocol.V4L2
        elif uri[:pos] == 'csi':
            protocol = Protocol.CSI
        elif uri[:pos] == 'rtsp':
            protocol = Protocol.RTSP
        else:
            protocol = Protocol.FILE
        return protocol
