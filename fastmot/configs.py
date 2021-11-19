from .detector import SSDDetector, YOLODetector, PublicDetector
from .tracker import MultiTracker, LPMOTTracker

def _default_detector(detector_class):
    def func(self, detector_cfg):
        return detector_class(self.size, self.class_ids, **vars(detector_cfg))
    return func

def _public_detector(self, detector_cfg):
    return PublicDetector(self.size, self.class_ids, self.detector_frame_skip, **vars(detector_cfg))


detectors = {
    'ssd': ('ssd_detector_cfg', _default_detector(SSDDetector)),
    'yolo': ('yolo_detector_cfg', _default_detector(YOLODetector)),
    'public': ('public_detector_cfg', _public_detector)
}

def _fastmot(self, tracker_cfg):
  # return MultiTracker(self.size, self.extractors[0].metric, **vars(tracker_cfg))
  return MultiTracker(self.size, 'cosine', **vars(tracker_cfg))

def _lpmot(self, tracker_cfg):
  return LPMOTTracker(self.size, self.extractors[0].metric, **vars(tracker_cfg))

trackers = {
    'fastmot': ('tracker_cfg', _fastmot),
    'lpmot': ('lpmot_tracker_cfg', _lpmot),
}
