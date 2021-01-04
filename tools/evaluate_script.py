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

from tools.pykitti_eval.kitti_eval import evaluate_kitti_mAP
from tools.visualizer.kitti_visual_tool import kitti_visual_tool_api


def pred_visualization(kitti_root, checkpoints_name, output_path, split="test"):
  image_path = os.path.join(kitti_root, "training/image_2")
  velodyne_path = os.path.join(kitti_root, "training/velodyne")
  calib_path = os.path.join(kitti_root, "training/calib")
  label_path = os.path.join(kitti_root, "training/label_2")
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
    label_2_file = os.path.join(label_path, base_name + ".txt")
    image = kitti_visual_tool_api(image_2_file, calib_file, label_2_file, velodyne_file)
    
    save_path = os.path.join(output_path, base_name)
    save_file = os.path.join(save_path, checkpoints_name)
    cv2.imsave(save_file, image)


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    checkpoints_path = "./checkpoints/DLA34_003nd"
    
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
        pred_label_path = os.path.join(cfg.OUTPUT_DIR, "inference", "kitti_train", "data")
        result_dict = evaluate_kitti_mAP(gt_label_path, pred_label_path, ["Car", "Pedestrian", "Cyclist"])
        
        if result_dict is not None:
            mAP_3d_moderate = result_dict["mAP3d"][1]
            val_mAP.append(mAP_3d_moderate)
            with open(os.path.join(cfg.OUTPUT_DIR, "val_mAP.json"),'w') as file_object:
                json.dump(val_mAP, file_object)
            with open(os.path.join(cfg.OUTPUT_DIR, 'epoch_result_{:07d}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
                f.write(result_dict["result"])
            print(result_dict["result"])

        if False:
            kitti_root = "datasets/kitti"
            checkpoints_name = model_name.split('.')[0]
            output_path = os.path.join(cfg.OUTPUT_DIR, "visualization")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            pred_visualization(kitti_root, checkpoints_name, output_path, "val")

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
