import collections
import random

import numpy as np

__all__ = ["bbox_iou", "box_random_crop_with_constraints", "box_flip", "box_resize", "box_translate", "collections", "random", "np"]

def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")
    # None 넣으면 axis 하나 더 생김
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2]) # (N,1,2) , (M,2) -> (N,M,2)
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4]) # (N,1,2) , (M,2) -> (N,M,2)
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2) # (N,M) * (N,M)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1) # (N,)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1) # (M,)
    return area_i / (area_a[:, None] + area_b - area_i) # (N,M) / (N,1) + (M,) - (N,M)


def box_random_crop_with_constraints(bbox, size, min_scale=0.1, max_scale=1,
                                     max_aspect_ratio=2, constraints=None,
                                     max_trial=50):
    """Crop an image randomly with bounding box constraints.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).

    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    # candidates = [(0, 0, w, h)] # stack 인 경우
    candidates = collections.deque([(0, 0, w, h)])

    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):

            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(1 / max_aspect_ratio, max_aspect_ratio)
            crop_h = int(h * np.sqrt(scale) / aspect_ratio)
            crop_w = int(w * np.sqrt(scale) * aspect_ratio)

            h_diff = h - crop_h
            w_diff = w - crop_w
            if h_diff < 1 or w_diff < 1:
                continue
            crop_t = random.randrange(h_diff)
            crop_l = random.randrange(w_diff)

            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right - left, bottom - top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right - left, bottom - top))
                break

    # random select one
    while candidates:
        # candidate = candidates.pop(np.random.randint(0, len(candidates))) # stack을 썼을 경우
        candidate = candidates.pop()
        new_bbox = box_crop(bbox, candidate, allow_outside_center=False)
        if new_bbox.size < 1:  # if not list(new_bbox) - 이거 더 느림
            continue
        return new_bbox, candidate
    return bbox, (0, 0, w, h)


def box_crop(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.

    This method is mainly used with image cropping to ensure bounding boxes fit
    within the cropped image.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.

    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders - 좌표 맞추기
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox


def box_resize(bbox, in_size, out_size):
    """Resize bouding boxes according to image resize operation.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    numpy.ndarray
        Resized bounding boxes with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    bbox = bbox.copy()
    x_scale = out_size[0] / in_size[0]
    y_scale = out_size[1] / in_size[1]

    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    return bbox


def box_flip(bbox, size, flip_x=False, flip_y=False):
    """Flip bounding boxes according to image flipping directions.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2: (width, height).
    flip_x : bool
        Whether flip horizontally.
    flip_y : type
        Whether flip vertically.

    Returns
    -------
    numpy.ndarray
        Flipped bounding boxes with original shape.
    """
    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))
    width, height = size
    bbox = bbox.copy()
    if flip_y:
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    if flip_x:
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax
    return bbox


def box_translate(bbox, x_offset=0, y_offset=0, shape=(512, 512)):
    """Translate bounding boxes by offsets.

    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    x_offset : int or float
        Offset along x axis.
    y_offset : int or float
        Offset along y axis.
    shape : image shape for limiting bbox size / h, w

    Returns
    -------
    numpy.ndarray
        Translated bounding boxes with original shape.
    """
    h, w = shape
    bbox = bbox.copy()
    bbox[:, :2] += (x_offset, y_offset)
    bbox[:, 2:4] += (x_offset, y_offset)
    bbox[:, 0] = np.clip(bbox[:, 0], 0, w)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, h)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, w)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, h)
    return bbox
