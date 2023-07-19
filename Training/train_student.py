import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
from PIL import Image
import numpy as np
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm
# from encoding.parallel import DataParallelModel, DataParallelCriterion
from ptsemseg.models.td4_psp.td4_psp import FCNHead, Layer_Norm, BatchNorm2d
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
# from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict,clean_state_dict
import pdb
import imageio
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

def train(cfg):
    # Setup seeds
    init_seed(1234, en_cudnn=False)

    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None)
    t_data_aug = get_composed_augmentations(train_augmentations)
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader

    path_n = cfg["model"]["path_num"]

    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(data_path,split=cfg["data"]["train_split"],augmentations=t_data_aug,path_num=path_n)
    # v_loader = data_loader(data_path,split=cfg["data"]["val_split"],augmentations=v_data_aug,path_num=path_n)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"],
                                  num_workers=cfg["training"]["n_workers"],
                                  shuffle=True,
                                  drop_last=True  )
    # valloader = data.DataLoader(v_loader,
    #                             batch_size=cfg["validating"]["batch_size"],
    #                             num_workers=cfg["validating"]["n_workers"] )
    
    # Setup Metrics
    running_metrics_val = runningScore(t_loader.n_classes)

    # Setup Model and Loss
    loss_fn = get_loss_function(cfg["training"])

    if cfg["training"]["finetune"] == True:
        nclass = 19
    else:
        nclass = 23

    model = get_model(cfg["model"], nclass,mdl_path = cfg["training"]["resume"], finetune=cfg["training"]["finetune"])
    state = torch.load(cfg["training"]["resume"])
    model.load_state_dict(state, strict=False)
    
    if cfg["training"]["finetune"] == True:
        nclass = 23
        norm_layer = BatchNorm2d
        
        model.layer_norm1 = Layer_Norm([32, 64])
        model.layer_norm2 = Layer_Norm([32, 64])
        model.layer_norm3 = Layer_Norm([32, 64])
        model.layer_norm4 = Layer_Norm([32, 64])

        model.head1 = FCNHead(512*model.expansion*1, nclass, norm_layer, chn_down=4)
        model.head2 = FCNHead(512*model.expansion*1, nclass, norm_layer, chn_down=4)
        model.head3 = FCNHead(512*model.expansion*1, nclass, norm_layer, chn_down=4)
        model.head4 = FCNHead(512*model.expansion*1, nclass, norm_layer, chn_down=4)

    model.cuda()

    # Setup optimizer
    optimizer = get_optimizer(cfg["training"], model)


    #Initialize training param
    cnt_iter = 0
    best_iou = 0.0
    time_meter = averageMeter()

    while cnt_iter <= cfg["training"]["train_iters"]:
        for (f_img, labels) in tqdm(trainloader):
            
            cnt_iter += 1
            # print(cnt_iter)
            
            model.train()
            optimizer.zero_grad()
            labels = labels.cuda()

            start_ts = time.time()
            outputs = model(f_img, pos_id=cnt_iter%path_n)

            seg_loss = loss_fn(outputs , labels)
            seg_loss.backward()

            time_meter.update(time.time() - start_ts)

            optimizer.step()

            if (cnt_iter + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                                            cnt_iter + 1,
                                            cfg["training"]["train_iters"],
                                            seg_loss.item(),
                                            time_meter.avg / cfg["training"]["batch_size"], )

                print(print_str)
                time_meter.reset()
                        
            if (cnt_iter + 1) % 1000 == 0:
                torch.save(model.state_dict(), f"checkpoint/finetune_viper_td4_psp18_{cnt_iter}.pth")

            
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    print("--- Start Fine Tuning The student ---")
    train(cfg)
