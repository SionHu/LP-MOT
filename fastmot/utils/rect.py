import numpy as np
import numba as nb


@nb.njit(cache=True)
def as_rect(tlbr):
    tlbr = np.asarray(tlbr, np.float64)
    tlbr = np.rint(tlbr)
    return tlbr


@nb.njit(cache=True)
def get_size(tlbr):
    tl, br = tlbr[:2], tlbr[2:]
    size = br - tl + 1
    return size


@nb.njit(cache=True)
def area(tlbr):
    size = get_size(tlbr)
    return int(size[0] * size[1])


@nb.njit(cache=True)
def mask_area(mask):
    return np.count_nonzero(mask)


@nb.njit(cache=True)
def get_center(tlbr):
    xmin, ymin, xmax, ymax = tlbr
    return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])


@nb.njit(cache=True)
def to_tlwh(tlbr):
    return np.append(tlbr[:2], get_size(tlbr))


@nb.njit(cache=True)
def to_tlbr(tlwh):
    tlwh = np.asarray(tlwh, np.float64)
    tlwh = np.rint(tlwh)
    tl, size = tlwh[:2], tlwh[2:]
    br = tl + size - 1
    return np.append(tl, br)


@nb.njit(cache=True)
def intersection(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    tl = np.maximum(tl1, tl2)
    br = np.minimum(br1, br2)
    tlbr = np.append(tl, br)
    if np.any(get_size(tlbr) <= 0):
        return None
    return tlbr


@nb.njit(cache=True)
def union(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    tl = np.minimum(tl1, tl2)
    br = np.maximum(br1, br2)
    tlbr = np.append(tl, br)
    return tlbr


@nb.njit(cache=True)
def crop(img, tlbr, chw=False):
    xmin, ymin, xmax, ymax = tlbr.astype(np.int_)
    if chw:
        return img[..., ymin:ymax + 1, xmin:xmax + 1]
    return img[ymin:ymax + 1, xmin:xmax + 1]


@nb.njit(cache=True)
def multi_crop(img, tlbrs):
    tlbrs_ = tlbrs.astype(np.int_)
    return [img[tlbrs_[i][1]:tlbrs_[i][3] + 1, tlbrs_[i][0]:tlbrs_[i][2] + 1]
            for i in range(len(tlbrs_))]


@nb.njit(fastmath=True, cache=True)
def iom(tlbr1, tlbr2):
    """
    Computes intersection over minimum.
    """
    tlbr = intersection(tlbr1, tlbr2)
    if tlbr is None:
        return 0.
    area_intersection = area(tlbr)
    area_minimum = min(area(tlbr1), area(tlbr2))
    return area_intersection / area_minimum


@nb.njit(fastmath=True, cache=True)
def transform(pts, m):
    """
    Numba implementation of OpenCV's transform.
    """
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1)
    return pts @ m.T


@nb.njit(fastmath=True, cache=True)
def perspective_transform(pts, m):
    """
    Numba implementation of OpenCV's perspectiveTransform.
    """
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T


@nb.njit(fastmath=True, cache=True)
def nms(tlwhs, scores, nms_thresh):
    """
    Applies Non-Maximum Suppression on the bounding boxes [x, y, w, h].
    Returns an array with the indexes of the bounding boxes we want to keep.
    """
    areas = tlwhs[:, 2] * tlwhs[:, 3]
    ordered = scores.argsort()[::-1]

    tl = tlwhs[:, :2]
    br = tlwhs[:, :2] + tlwhs[:, 2:] - 1

    keep = []
    while ordered.size > 0:
        # index of the current element
        i = ordered[0]
        keep.append(i)

        other_tl = tl[ordered[1:]]
        other_br = br[ordered[1:]]

        # compute IoU
        inter_xmin = np.maximum(tl[i, 0], other_tl[:, 0])
        inter_ymin = np.maximum(tl[i, 1], other_tl[:, 1])
        inter_xmax = np.minimum(br[i, 0], other_br[:, 0])
        inter_ymax = np.minimum(br[i, 1], other_br[:, 1])

        inter_w = np.maximum(0, inter_xmax - inter_xmin + 1)
        inter_h = np.maximum(0, inter_ymax - inter_ymin + 1)
        inter_area = inter_w * inter_h
        union_area = areas[i] + areas[ordered[1:]] - inter_area
        iou = inter_area / union_area

        idx = np.where(iou <= nms_thresh)[0]
        ordered = ordered[idx + 1]
    keep = np.asarray(keep)
    return keep


@nb.njit(fastmath=True, cache=True)
def diou_nms(tlwhs, scores, nms_thresh, beta=0.6):
    """
    Applies Non-Maximum Suppression using the DIoU metric.
    """
    areas = tlwhs[:, 2] * tlwhs[:, 3]
    ordered = scores.argsort()[::-1]

    tl = tlwhs[:, :2]
    br = tlwhs[:, :2] + tlwhs[:, 2:] - 1
    centers = (tl + br) / 2

    keep = []
    while ordered.size > 0:
        # index of the current element
        i = ordered[0]
        keep.append(i)

        other_tl = tl[ordered[1:]]
        other_br = br[ordered[1:]]

        # compute IoU
        inter_xmin = np.maximum(tl[i, 0], other_tl[:, 0])
        inter_ymin = np.maximum(tl[i, 1], other_tl[:, 1])
        inter_xmax = np.minimum(br[i, 0], other_br[:, 0])
        inter_ymax = np.minimum(br[i, 1], other_br[:, 1])

        inter_w = np.maximum(0, inter_xmax - inter_xmin + 1)
        inter_h = np.maximum(0, inter_ymax - inter_ymin + 1)
        inter_area = inter_w * inter_h
        union_area = areas[i] + areas[ordered[1:]] - inter_area
        iou = inter_area / union_area

        # compute DIoU
        union_xmin = np.minimum(tl[i, 0], other_tl[:, 0])
        union_ymin = np.minimum(tl[i, 1], other_tl[:, 1])
        union_xmax = np.maximum(br[i, 0], other_br[:, 0])
        union_ymax = np.maximum(br[i, 1], other_br[:, 1])

        union_w = union_xmax - union_xmin + 1
        union_h = union_ymax - union_ymin + 1
        c = union_w**2 + union_h**2
        d = np.sum((centers[i] - centers[ordered[1:]])**2, axis=1)
        diou = iou - (d / c)**beta

        idx = np.where(diou <= nms_thresh)[0]
        ordered = ordered[idx + 1]
    keep = np.asarray(keep)
    return keep
