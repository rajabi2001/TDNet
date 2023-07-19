import os
import torch
import numpy as np
import imageio
import cv2
from tqdm import tqdm

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)]


class cityscapesLoader():

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))


    def __init__(self,img_path,in_size):
        self.img_path = img_path
        self.n_classes = 19
        self.files = recursive_glob(rootdir=self.img_path, suffix=".png")
        self.files.sort()
        self.files_num = len(self.files)
        self.data = []
        self.size = (in_size[1],in_size[0])
        self.mean = np.array([.485, .456, .406])
        self.std = np.array([.229, .224, .225])

    def load_frames(self):

        for idx in range(self.files_num):
            img_path = self.files[idx].rstrip()
            img_name = img_path.split('/')[-1]
            folder = img_path.split('/')[-2]

            #img = cv2.imread(img_path).astype(np.float32)
            img = imageio.imread(img_path)
            ori_size = img.shape[:-1]

            img = cv2.resize(img,self.size)/255.0
            img = (img-self.mean)/self.std

            img = img.transpose(2, 0, 1)
            img = img[np.newaxis,:]
            img = torch.from_numpy(img).float()

            self.data.append([img,img_name,folder,self.size])

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r #/ 255.0
        rgb[:, :, 1] = g #/ 255.0
        rgb[:, :, 2] = b #/ 255.0
        return rgb

class Viper():

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
        # [0, 0, 0],
        [111, 74, 0],
        [70, 130, 180],
        [128, 64, 128],
        [244, 35, 232],
        [230, 150, 140],
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
        [255, 0, 0],
        [119, 11, 32],
        [0, 0, 230],
        [0, 0, 142],
        [0, 80, 100],
        [0, 60, 100],
        [0, 0, 70],
        [0, 0, 90],
        [0, 80, 100],
        [0, 100, 100],
        [50, 0, 90]
    ]

    label_colours = dict(zip(range(31), colors))

    def __init__(self,img_path,in_size):
        self.img_path = img_path
        self.n_classes = 31
        self.files = recursive_glob(rootdir=self.img_path, suffix=".jpg")
        self.files.sort()
        self.files_num = len(self.files)
        self.data = []
        self.size = (in_size[1],in_size[0])
        self.mean = np.array([.485, .456, .406])
        self.std = np.array([.229, .224, .225])

    def load_frames(self):

        for idx in tqdm(range(self.files_num)):
            img_path = self.files[idx].rstrip()
            # img_name = img_path.split('/')[-1]
            # folder = img_path.split('/')[-2]

            #img = cv2.imread(img_path).astype(np.float32)
            img = imageio.imread(img_path)
            ori_size = img.shape[:-1]

            img = cv2.resize(img,self.size)/255.0
            img = (img-self.mean)/self.std

            img = img.transpose(2, 0, 1)
            img = img[np.newaxis,:]
            img = torch.from_numpy(img).float()

            # self.data.append([img,img_name,folder,self.size])
            self.data.append([img,self.size])

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r #/ 255.0
        rgb[:, :, 1] = g #/ 255.0
        rgb[:, :, 2] = b #/ 255.0
        return rgb

















	
        
