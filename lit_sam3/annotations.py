"""Full-video annotation loading for the original LIT evaluation workflow."""

from pathlib import Path

import numpy as np
from PIL import Image


def _read_mask(path):
    with Image.open(path) as image:
        return np.array(image)


def load_object_annotations(mask_dir, frame_names, object_ids, per_obj_png_file=False):
    """Load all annotations and retain each object's masks in CPU memory.

    Packed annotations match frame stems. Separate-object annotations retain
    the evaluation's sparse, zero-based frame-index convention.
    """
    mask_dir = Path(mask_dir)
    if not per_obj_png_file:
        masks = [_read_mask(mask_dir / f"{name}.png") for name in frame_names]
        return {
            obj: [np.where(mask == obj, 1, np.where(mask == 255, 255, 0)).astype(mask.dtype)
                  for mask in masks]
            for obj in object_ids
        }
    directories = {int(p.name): p for p in mask_dir.iterdir() if p.is_dir()}
    result = {}
    for obj in object_ids:
        indexed = {int(p.stem): _read_mask(p) > 0 for p in directories[obj].glob('*.png')}
        result[obj] = [indexed.get(i) for i in range(len(frame_names))]
    return result
