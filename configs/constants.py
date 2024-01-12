'''
MAP_FROM_MAVIS_TO_TRAIN = {**{
    0: 0,
    1: 9,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 2,
    7: 4,
    8: 2,
    9: 5,
    10: 10,
    11: 1,
    12: 6,
    13: 2,
    14: 1,
    15: 1,
    16: 7,
    17: 9,
    18: 4,
    19: 8,}, **{i:9 for i in range(64, 101)}, **{j:10 for j in range(101, 181)}} 
# 101 to 180 are person, and 64 to 100 vehicle
'''

'''
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
'''

MAP_FROM_MAVIS_TO_TRAIN = {**{
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 9,
    7: 4,
    8: 2,
    9: 5,
    10: 1,
    11: 8,
    12: 6,
    13: 2,
    14: 1,
    15: 1,
    16: 7,
    17: 1,
    18: 4,
    19: 2,
    20: 0,}, **{i:1 for i in range(21, 64)}, **{i:1 for i in range(64, 181)}}

MAP_FROM_MAVIS_TO_TRAIN_BATCH9 = {**{
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 9,
    6: 4,
    7: 2,
    8: 5,
    9: 8,
    10: 6,
    11: 2,
    12: 1,
    13: 1,
    14: 7,
    15: 1,
    16: 4,
    17: 2,
    18: 0,
    19: 0,}, **{i:1 for i in range(20, 181)}}

FROM_CATEGORY = ["rgb", "id"]
AIM_CATEGORY = ["rgb_1024", "id_1024"]  # ['labels', 'images']
SUPPORT_FORMAT = [".png", ".jpg"]
INDEX_PHASES = [7, 8, 10]
IGNORE_CLASS = 0
AIM_SIZE = (1024, 1024)

'''
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
    9: (160, 160, 160), # vehicle - grey 
    10: (153, 225, 225),  # person -- light yellow
    # 11: (255, 102, 178), # animal -- purple
    # 12: (40, 40, 230), # implement -- bright_red
}  # bgr
'''

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
    9: (170, 100, 60)
}  # bgr

IMG_DIR_PUT_MODEL = "/data/segmentation/zlin_data/pidnet_dataset/rgb_1024"  # for personal dataset 
LABEL_DIR_PUT_MODEL = "/data/segmentation/zlin_data/pidnet_dataset/id_1024_sky_updated"
BATCHES = [
    # "/data/segmentation/zlin_data/pidnet_dataset/coco",
    # "/data/segmentationcd/zlin_data/pidnet_dataset/cityscapes",
    # "/data/segmentation/zlin_data/pidnet_dataset/cityscapes_for_pidnet",
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

AUG_HUMAN_PATH =  '/data/segmentation/zlin_data/pidnet_dataset/aug_human' 
AUG_VEHICLE_PATH = '/data/segmentation/zlin_data/pidnet_dataset/aug_vehicle'

HUMAN_CLASS = 1 #10
VEHICLE_CLASS = 1 #9
