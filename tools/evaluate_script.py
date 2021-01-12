import os
import torch
import json

from smoke.config import cfg
from smoke.data import make_data_loader
from smoke.solver.build import (
    make_optimizer,
    make_lr_scheduler,
)
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from smoke.utils import comm
from smoke.modeling.detector import build_detection_model
from smoke.engine.test_net import run_test

from tools.evaluation.kitti_utils.eval import kitti_eval
from tools.evaluation.kitti_utils import kitti_common as kitti

from tools.visualizer.kitti_visual_tool import kitti_visual_tool_api

import cv2


def pred_visualization(kitti_root, checkpoints_name, output_path, split="test"):
  image_path = os.path.join(kitti_root, "training/image_2")
  velodyne_path = os.path.join(kitti_root, "training/velodyne")
  calib_path = os.path.join(kitti_root, "training/calib")
  label_path = os.path.join(kitti_root, "training/label_2")
  pred_path = os.path.join(output_path, "inference", "kitti_train", "data")

  output_path = os.path.join(output_path, "visualization")
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  if split == "train":
    imageset_txt = os.path.join(kitti_root, "ImageSets", "train.txt")
  elif split == "val":
    imageset_txt = os.path.join(kitti_root, "ImageSets", "val.txt")
  elif split == "test":
    imageset_txt = os.path.join(kitti_root, "ImageSets", "test.txt")
  elif split == "visual":
    imageset_txt = os.path.join(kitti_root, "ImageSets", "visual.txt")
  else:
    raise ValueError("Invalid split!")

  image_ids = []
  for line in open(imageset_txt, "r"):
    base_name = line.replace("\n", "")
    if not os.path.exists(os.path.join(output_path, base_name)):
      os.makedirs(os.path.join(output_path, base_name))
    image_ids.append(base_name)

  for i in range(len(image_ids)):
    base_name = image_ids[i]
    image_2_file = os.path.join(image_path, base_name + ".png")
    velodyne_file = os.path.join(velodyne_path, base_name + ".bin")
    calib_file = os.path.join(calib_path, base_name + ".txt")
    label_2_file = os.path.join(pred_path, base_name + ".txt")
    image = kitti_visual_tool_api(image_2_file, calib_file, label_2_file, velodyne_file)
    
    save_path = os.path.join(output_path, base_name)
    save_file = os.path.join(save_path, checkpoints_name) + ".jpg"
    print(save_file)
    cv2.imwrite(save_file, image)


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    checkpoints_path = cfg.OUTPUT_DIR
    
    val_mAP = []
    iteration_list = []
    for model_name in os.listdir(checkpoints_path):
        if "pth" not in model_name or "final" in model_name:
            continue
        iteration = int(model_name.split(".")[0].split('_')[1])
        iteration_list.append(iteration)
    iteration_list = sorted(iteration_list)
    
    for iteration in iteration_list:
        
        model = build_detection_model(cfg)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        
        model_name = "model_{:07d}.pth".format(iteration)
        ckpt = os.path.join(checkpoints_path, model_name)
        _ = checkpointer.load(ckpt, use_latest=False)
        run_test(cfg, model)
        
        gt_label_path = "datasets/kitti/training/label_2/"
        pred_label_path = os.path.join(checkpoints_path, "inference", "kitti_train", "data")

        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"])

        if ret_dict is not None:
            mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
            val_mAP.append(mAP_3d_moderate)
            with open(os.path.join(checkpoints_path, "val_mAP.json"),'w') as file_object:
                json.dump(val_mAP, file_object)
            with open(os.path.join(checkpoints_path, 'epoch_result_{:07d}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
                f.write(result)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
