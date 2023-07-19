import os
import torch
import numpy as np
import imageio
from PIL import Image
from ptsemseg.utils import recursive_glob
from tqdm import tqdm
from numba import cuda, jit
import cv2

rgb2id = {
    (0, 0, 0): (0, "unlabeled"),
    (111, 74, 0): (1, "ambiguous"),  
    (70, 130, 180): (2, "sky"), 
    (128, 64, 128): (3, "road"), 
    (244, 35, 232): (4, "sidewalk"), 
    (230, 150, 140): (5, "railtrack"),
    (152, 251, 152): (6, "terrain"), 
    (87, 182, 35): (7, "tree"), 
    (35, 142, 35): (8, "vegetation"), 
    (70, 70, 70): (9, "building"), 
    (153, 153, 153): (10, "infrastructure"), 
    (190, 153, 153): (11, "fence"), 
    (150, 20, 20): (12, "billboard"), 
    (250, 170, 30): (13, "traffic light"), 
    (220, 220, 0): (14, "traffic sign"), 
    (180, 180, 100): (15, "mobilebarrier"), 
    (173, 153, 153): (16, "firehydrant"),
    (168, 153, 153): (17, "chair"),
    (81, 0, 21): (18, "trash"),
    (81, 0, 81): (19, "trashcan"),
    (220, 20, 60): (20, "person"),
    (255, 0, 0): (21, "animal"),
    (119, 11, 32): (22, "bicycle"),
    (0, 0, 230): (23, "motorcycle"),
    (0, 0, 142): (24, "car"),
    (0, 80, 100): (25, "van"),
    (0, 60, 100): (26, "bus"),
    (0, 0, 70): (27, "truck"),
    (0, 0, 90): (28, "trailer"),
    (0, 80, 100): (29, "train"),
    (0, 100, 100): (30, "plane"),
    (50, 0, 90): (31, "boat"),
}
id2rgb = {
    0 : (0, 0, 0),
    1 : (111, 74, 0), 
    2 : (70, 130, 180), 
    3 : (128, 64, 128),
    4 : (244, 35, 232), 
    5 : (230, 150, 140),
    6 : (152, 251, 152),
    7 : (87, 182, 35),
    8 : (35, 142, 35), 
    9 : (70, 70, 70), 
    10 : (153, 153, 153),
    11 : (190, 153, 153),
    12 : (150, 20, 20),
    13 : (250, 170, 30), 
    14 : (220, 220, 0),
    15 : (180, 180, 100),
    16 : (173, 153, 153),
    17 : (168, 153, 153),
    18 : (81, 0, 21),
    19 : (81, 0, 81),
    20 : (220, 20, 60), 
    21 : (255, 0, 0),
    22 : (119, 11, 32),
    23 : (0, 0, 230),
    24 : (0, 0, 142),
    25 : (0, 80, 100),
    26 : (0, 60, 100),
    27 : (0, 0, 70),
    28 : (0, 0, 90),
    29 : (0, 80, 100),
    30 : (0, 100, 100), 
    31 : (50, 0, 90),
}
colors = [
    [70, 130, 180],
    [128, 64, 128],
    [244, 35, 232],
    [152, 251, 152],
    [87, 182, 35],
    [35, 142, 35], 
    [70, 70, 70], 
    [153, 153, 153],
    [190, 153, 153],
    [150, 20, 20],
    [250, 170, 30], 
    [220, 220, 0],
    [180, 180, 100],
    [173, 153, 153],
    [168, 153, 153],
    [81, 0, 21],
    [81, 0, 81],
    [220, 20, 60], 
    [0, 0, 230],
    [0, 0, 142],
    [0, 80, 100],
    [0, 60, 100],
    [0, 0, 70],
]

void_classes = [0, 1, 5, 21, 22, 28, 29, 30, 31]
valid_classes = [2, 3, 4, 6, 7, 8, 9, 10 ,11 ,12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27]

ignore_index = 250
class_map = dict(zip(valid_classes, range(23)))
 
def encode_segmap(lbl):

    lbl_gray = cv2.cvtColor(lbl,cv2.COLOR_BGR2GRAY)
    fake_lbl = np.ones_like(lbl_gray)*ignore_index

    for class_id,color in id2rgb.items() :
            r, g, b = color
            gray = round(0.299*b+0.587*g+0.114*r)
            if class_id not in void_classes:
                fake_lbl[lbl_gray[:,:] == gray] = class_map[class_id]

    return fake_lbl

def make_same_directories(root, split):
    s_path = f"{root}/{split}/cls/"
    d_path = f"{root}/{split}/cls_encoded/"
    for i in os.listdir(s_path):
        os.mkdir(d_path+i)

if __name__ == "__main__":

    print("hi")

    root="data/viper"
    split="train"

    make_same_directories(root, split)
    
    fileslbl = {}
    files = {}
    annotations_base = os.path.join(root, split, "cls")
    images_base = os.path.join(root, split, "img")
    fileslbl[split] = recursive_glob(rootdir=annotations_base, suffix=".png")
    files[split] = recursive_glob(rootdir=images_base, suffix=".jpg")

    # # checking 
    # for i in tqdm(range(len(files[split]))):
    #     img_path = files[split][i].rstrip()
    #     img = imageio.imread(img_path)

    cntr = 0

    for i in tqdm(range(len(fileslbl[split]))):

        # if i < cntr :
        #     continue

        lbl_path = fileslbl[split][i].rstrip()
        lbl = imageio.imread(lbl_path, pilmode="RGB")
        lbl = encode_segmap(np.array(lbl, dtype=np.uint8))
        lbl_path = lbl_path.replace("cls","cls_encoded")
        imageio.imwrite(lbl_path, lbl)


