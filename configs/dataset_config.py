import os


DATA_ROOT = "/opt/data/private/data/LIT_Data"


DATASET_CONFIGS = {
    "lvosv2": {
        "root": os.path.join(DATA_ROOT, "LVOSv2", "valid"),
        "image_dir": "JPEGImages",
        "annotation_dir": "Annotations",
        "video_list_file": None,
        "per_obj_png_file": False,
        "track_object_appearing_later_in_video": True,
    },

    "sav_val": {
        "root": os.path.join(DATA_ROOT, "SA-V", "sav_val"),
        "image_dir": "JPEGImages_24fps",
        "annotation_dir": "Annotations_6fps",
        "video_list_file": os.path.join(DATA_ROOT, "SA-V", "sav_val", "sav_val.txt"),
        "per_obj_png_file": True,
        "track_object_appearing_later_in_video": False,
    },

    "sav_test": {
        "root": os.path.join(DATA_ROOT, "SA-V", "sav_test"),
        "image_dir": "JPEGImages_24fps",
        "annotation_dir": "Annotations_6fps",
        "video_list_file": os.path.join(DATA_ROOT, "SA-V", "sav_test", "sav_test.txt"),
        "per_obj_png_file": True,
        "track_object_appearing_later_in_video": False,
    },

    "vost": {
    "root": os.path.join(DATA_ROOT, "VOST"),
    "image_dir": "JPEGImages",
    "annotation_dir": "Annotations",
    "video_list_file": os.path.join(DATA_ROOT, "VOST", "ImageSets", "val.txt"),
    "per_obj_png_file": False,
    "track_object_appearing_later_in_video": False,
},
}


def get_dataset_config(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIGS.keys())}")

    cfg = DATASET_CONFIGS[dataset_name].copy()
    cfg["base_video_dir"] = os.path.join(cfg["root"], cfg.pop("image_dir"))
    cfg["input_mask_dir"] = os.path.join(cfg["root"], cfg.pop("annotation_dir"))

    return cfg