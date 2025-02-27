import os
import torch
import argparse
import timeit
import numpy as np
import oyaml as yaml
from torch.utils import data
from PIL import Image
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.loss import get_loss_function
import cv2
import pdb

torch.backends.cudnn.benchmark = True

print(torch.__version__)
def validate(cfg, args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    path_n = cfg["model"]["path_num"]

    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    v_loader = data_loader(
        data_path,
        # split=cfg["data"]["val_splitimage.png"],
        split=cfg["data"]["train_split"],
        # img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=v_data_aug,
        path_num=path_n
    )

    n_classes = v_loader.n_classes
    # n_classes = 23
    valloader = data.DataLoader(
        v_loader, batch_size=cfg["validating"]["batch_size"], num_workers=cfg["validating"]["n_workers"]
    )

    # running_metrics = runningScore(n_classes)
    # loss_fn = get_loss_function(cfg["training"])

    # Setup Model
    # teacher = get_model(cfg["teacher"], n_classes)
    model = get_model(cfg["model"], n_classes,mdl_path = cfg["training"]["resume"]).to(device)
    state = torch.load(cfg["validating"]["resume"])
    model.load_state_dict(state, strict=False)
    print("Initialized sub networks with pretrained '{}'".format(cfg["validating"]["resume"]))
    model.eval()
    model.to(device)

    model_tag = str(cfg["validating"]["resume"]).split('/')[2].split('.')[0].split("_")[4]
    path_result = f"result/{model_tag}"
    if not os.path.isdir(path_result):
        os.makedirs(path_result)

    with torch.no_grad():
        for i, (val, labels) in enumerate(valloader):

            # gt = labels.numpy()
            _val = [ele.to(device) for ele in val]


            # torch.cuda.synchronize()
            # start_time = timeit.default_timer()
            outputs = model(_val,pos_id=i%path_n)
            # torch.cuda.synchronize()
            # elapsed_time = timeit.default_timer() - start_time
            pred = outputs.data.max(1)[1].cpu().numpy()
            # running_metrics.update(gt, pred)
                      
            
            # if args.measure_time:
            #     elapsed_time = timeit.default_timer() - start_time
            #     print(
            #         "Inference time \
            #           (iter {0:5d}): {1:3.5f} fps".format(
            #             i + 1, pred.shape[0] / elapsed_time
            #         )
            #     )

            if True:
                
                decoded = v_loader.decode_segmap(pred[0])
                path_save_result = path_result + f"/frame_{i}.png"
                Image.fromarray(np.uint8(decoded)).convert('RGB').save(path_save_result)

                # cv2.namedWindow("Image")
                # cv2.imshow("Image", decoded)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()



    # score, class_iou = running_metrics.get_scores()

    # for k, v in score.items():
    #     print(k, v)

    # for i in range(n_classes):
    #     print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--gpu",
        nargs="?",
        type=str,
        default="0",
        help="GPU ID",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )


    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    validate(cfg, args)
