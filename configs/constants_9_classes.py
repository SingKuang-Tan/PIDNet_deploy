MAP_FROM_MAVIS_TO_TRAIN = {**{
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 2,
    7: 4,
    8: 2,
    9: 5,
    10: 1,
    11: 1,
    12: 6,
    13: 2,
    14: 1,
    15: 1,
    16: 7,
    17: 1,
    18: 4,
    19: 8,}, **{i:1 for i in range(64, 181)}} 
# 101 to 180 are person, and 64 to 100 vehicle


FROM_CATEGORY = ["rgb", "id"]
AIM_CATEGORY = ["rgb_1024", "id_1024"]  # ['labels', 'images']
SUPPORT_FORMAT = [".png", ".jpg"]
INDEX_PHASES = [7, 8, 10]
IGNORE_CLASS = 0
AIM_SIZE = (1024, 1024)

LABEL_TO_COLOR = {
    0: (0, 0, 0), # black
    1: (246, 4, 228), # obstacle pink - red
    2: (173, 94, 48), # navigable space - dark blue
    3: (68, 171, 117), # vine_canopy green - green
    4: (162, 122, 174), # vine_stem - pink
    5: (121, 119, 148), # wine_pole - yellow-white
    6: (253, 75, 40), # vegetarian - light blue
    7: (170, 60, 100), # hood 
    8: (60, 100, 179), # sky orange
    # 9: (170, 100, 60)
}  # bgr

IMG_DIR_PUT_MODEL = "/data/segmentation/zlin_data/pidnet_dataset/rgb_1024"  # for personal dataset 
LABEL_DIR_PUT_MODEL = "/data/segmentation/zlin_data/pidnet_dataset/id_1024"
BATCHES = [
    # "/data/segmentation/zlin_data/pidnet_dataset/coco",
    # "/data/segmentationcd/zlin_data/pidnet_datasetcityscapes",
    "/data/segmentation/Batch8",
    "/data/segmentation/Batch3x",
    "/data/segmentation/Batch1x",
    "/data/segmentation/Batch2x",
    "/data/segmentation/Batch4",
    "/data/segmentation/Batch5",
    "/data/segmentation/Batch6",
    "/data/segmentation/Batch7",
]


SEQUENCE_LIST_ROOT = '/data/segmentation/zlin_data/pidnet_dataset/sequence_dataset/sequence/'
SEQUENCE_FILE_LIST = '/data/segmentation/zlin_data/pidnet_dataset/sequence_dataset/list.txt'
LDOF_FILE_PATH = '/data/segmentation/zlin_data/pidnet_dataset/sequence_dataset/ldof_flow/'
AUG_HUMAN_PATH = '/data/segmentation/zlin_data/pidnet_dataset/aug_human'
AUG_VEHICLE_PATH = '/data/segmentation/zlin_data/pidnet_dataset/aug_vehicle'

HUMAN_CLASS = 10
alfa = 0.2 