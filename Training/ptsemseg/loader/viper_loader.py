import os
import torch
import numpy as np
import imageio
from PIL import Image
from torch.utils import data
import random
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from torchvision.utils import save_image
from tqdm import tqdm
import cv2


class ViperLoader(data.Dataset):
    
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

    label_colours = dict(zip(range(23), colors))

    def __init__(
        self,
        root="data/viper",
        split="train",
        augmentations=None,
        test_mode=False,
        model_name=None,
        interval=2,
        path_num=2,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num= path_num
        self.interval = interval
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.test_mode=test_mode
        self.model_name=model_name
        self.n_classes = 23
        self.files = {}
        self.fileslbl = {}

        self.images_base = os.path.join(self.root, self.split, "img")

        
        # self.annotations_base = os.path.join(self.root, self.split, "cls_encoded")
        # self.annotations_base = os.path.join(self.root, self.split, "cls")


        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")
        # self.fileslbl[split] = recursive_glob(rootdir=self.annotations_base, suffix=".png")

        self.void_classes = [0, 1, 5, 21, 22, 28, 29, 30, 31]
        self.valid_classes = [2, 3, 4, 6, 7, 8, 9, 10 ,11 ,12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27]
        
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(23)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if not self.test_mode:
            img_path = self.files[self.split][index].rstrip()
            # lbl_path = self.fileslbl[self.split][index].rstrip()

                
            vid_info = img_path.replace('\\','/').split('/')[-1].split('_')
            folder, frame = vid_info[0], vid_info[1].split('.')[0]
            

            f4_id = int(frame)
            f3_id = f4_id - 1
            f2_id = f3_id - 1
            f1_id = f2_id - 1

            f4_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f4_id)))
            f4_img = imageio.imread(f4_path)
            f4_img = np.array(f4_img, dtype=np.uint8)

            f3_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f3_id)))
            if not os.path.isfile(f3_path):
                f3_path = f4_path
            f3_img = imageio.imread(f3_path)
            f3_img = np.array(f3_img, dtype=np.uint8)
            
            f2_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f2_id)))
            if not os.path.isfile(f2_path):
                f2_path = f4_path
            f2_img = imageio.imread(f2_path)
            f2_img = np.array(f2_img, dtype=np.uint8)

            f1_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f1_id)))
            if not os.path.isfile(f1_path):
                f1_path = f4_path
            f1_img = imageio.imread(f1_path)
            f1_img = np.array(f1_img, dtype=np.uint8)

            lbl_path = f4_path.replace("img","cls_encoded").replace("jpg","png")
            lbl = imageio.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.uint8)

            if self.augmentations is not None:
                [f4_img, f3_img, f2_img, f1_img], lbl = self.augmentations([f4_img, f3_img, f2_img, f1_img], lbl)

            f4_img = f4_img.float()
            f3_img = f3_img.float()
            f2_img = f2_img.float()
            f1_img = f1_img.float()
            lbl = torch.from_numpy(lbl).long()

            if self.path_num == 4:
                return [f1_img, f2_img, f3_img, f4_img], lbl
            else:
                return [f3_img, f4_img], lbl


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        # rgb[:, :, 0] = b / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = r / 255.0
        return rgb

    
    def encode_segmap(self, lbl):

        lbl_gray = cv2.cvtColor(lbl,cv2.COLOR_BGR2GRAY)
        fake_lbl = np.ones_like(lbl_gray)*self.ignore_index

        for class_id,color in self.id2rgb.items() :
                r, g, b = color
                gray = round(0.299*b+0.587*g+0.114*r)
                if class_id not in self.void_classes:
                    fake_lbl[lbl_gray[:,:] == gray] = self.class_map[class_id]

        return fake_lbl


    def decode_pred(self, lbl):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            lbl[lbl == _predc] = self.valid_classes[_predc]
        return lbl.astype(np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/data/viper/"
    dst = ViperLoader(local_path, is_transform=True)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        import pdb

        pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
