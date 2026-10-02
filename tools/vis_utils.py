
import numpy as np
from pycocotools import mask as MaskUtils
import json, cv2, colorsys
import math
# from skimage.morphology import disk


def generate_rand_colors(num_colors, seed=0, lightness=1, saturation=1):
    """ Generate a list of random colors in RGB format.
    """
    uniform_colors = [colorsys.hsv_to_rgb(i / num_colors, saturation, lightness) for i in range(num_colors)]
    uniform_colors = np.array(uniform_colors) * 255

    np.random.seed(seed)
    np.random.shuffle(uniform_colors)

    return uniform_colors

def apply_anno(image, prompts=None, mask=None, mask_color=0, mask_alpha=0.3, contour_color=None, contour_thickness=2):
    """ Apply annotations to an image.    
    """
    if prompts is not None:
        coords, labels = prompts['points'], prompts['labels']
        for point in coords[labels==1]:
            cv2.drawMarker(image, tuple(point.astype(int)), color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=17, thickness=5, line_type=cv2.LINE_AA)
            cv2.drawMarker(image, tuple(point.astype(int)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
        for point in coords[labels==0]:
            cv2.drawMarker(image, tuple(point.astype(int)), color=(255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=17, thickness=5, line_type=cv2.LINE_AA)
            cv2.drawMarker(image, tuple(point.astype(int)), color=(0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2, line_type=cv2.LINE_AA)

    if mask is not None:
        if type(mask_color) == int:
            mask_color = generate_rand_colors(10, lightness=1, seed=10)[mask_color % 10]
        
        if contour_color is None:
            contour_color = mask_color
        elif type(contour_color) == int:
            contour_color = generate_rand_colors(10, lightness=1, seed=10)[contour_color % 10]
        
        image[mask] = (image[mask].astype(float) * (1 - mask_alpha) + mask_alpha * mask_color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color=(contour_color[0], contour_color[1], contour_color[2]), thickness=contour_thickness, lineType=cv2.LINE_AA)

    return image

def rle_to_bmask(rle_mask):
    rle = [{'counts': rle_mask['counts'].encode('ascii'), 'size': rle_mask['size']}]
    return MaskUtils.decode(rle)[...,-1].astype(bool)

def bmask_to_rle(binary_mask):
    assert binary_mask.dtype == np.bool_, "Expecting binary mask"
    assert binary_mask.ndim == 2, "Expecting 2D mask"

    rle = MaskUtils.encode(np.asfortranarray(binary_mask))
    return {'counts': rle['counts'].decode('ascii'),
            'size': rle['size']}

def apply_func_to_leaves(data, function=None):
    """
    Recursively applies a function to all leaf values in a nested data structure.
    """

    if isinstance(data, dict):
        return {k: apply_func_to_leaves(v, function) for k, v in data.items()}
    elif isinstance(data, list):
        return [apply_func_to_leaves(v, function) for v in data]
    elif isinstance(data, tuple):
        return tuple(apply_func_to_leaves(v, function) for v in data)
    elif isinstance(data, set):
        # Note: sets can only contain hashable elements, so they may not work for all nested structures
        return {apply_func_to_leaves(v, function) for v in data}
    else:
        # This is a leaf (non-container value) - transform it if a function is provided
        return function(data) if function is not None else data


def get_iou(obj_mask, obj_gt, void_pixel=255):
    obj_void = obj_gt == void_pixel
    obj_void = ~obj_void
    obj_gt = (obj_gt > 0) & (obj_gt != void_pixel)
    intersection = (obj_mask * obj_gt * obj_void).sum()
    pixel_sum = (obj_mask * obj_void).sum() + (obj_gt * obj_void).sum()
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1
    return intersection / (pixel_sum - intersection)

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Cannot convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def get_f(obj_mask, obj_gt, boundary=0.008):

    # boundary disk for boundary F-score. It is the same for all objects.
    bound_pix = np.ceil(boundary * np.linalg.norm(obj_mask.shape))
    boundary_disk = disk(bound_pix)

    mask_boundary = _seg2bmap(obj_mask)
    gt_boundary = _seg2bmap(obj_gt)
    mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

    # Get the intersection
    gt_match = gt_boundary * mask_dilated
    fg_match = mask_boundary * gt_dilated

    # Area of the intersection
    n_fg = np.sum(mask_boundary)
    n_gt = np.sum(gt_boundary)

    # Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    return F