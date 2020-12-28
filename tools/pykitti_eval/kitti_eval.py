import os
import numpy as np
from tools.pykitti_eval import kitti_common as kitti
from tools.pykitti_eval.eval import get_coco_eval_result, get_official_eval_result

def evaluate_kitti_mAP(gt_label_path, pred_label_path, class_name = ["Car", "Pedestrian", "Cyclist"]):
  if not os.path.exists(gt_label_path):
    print("gt_label_path not found")
  if not os.path.exists(pred_label_path):
    print("pred_label_path not found")

  pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
  gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
  print("pred_annos: ", len(pred_annos))
  print("gt_annos: ", len(gt_annos))
  result_dict = dict()
  if len(pred_annos) > 0:
    result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(gt_annos, pred_annos, class_name, return_data=True)

    result_dict["mAPbbox"] = mAPbbox[0, :, 0] if mAPbbox is not None else np.zeros((3))
    result_dict["mAPbev"] = mAPbev[0, :, 0] if mAPbev is not None else np.zeros((3))
    result_dict["mAP3d"] = mAP3d[0, :, 0] if mAP3d is not None else np.zeros((3))
    result_dict["mAPaos"] = mAPaos[0, :, 0] if mAPaos is not None else np.zeros((3))
    result_dict["result"] = result if result is not None else ""

  return result_dict
