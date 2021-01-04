import argparse
import os
import time
import csv
import cv2
import numpy as np
from PIL import Image

from kitti_utils import *
from utils import (
  load_intrinsic, compute_box_3d_camera, project_to_image, 
  draw_box_3d, draw_3d_box_on_image, draw_box_on_bev_image)

import warnings
warnings.filterwarnings("ignore")

def kitti_visual_tool_api(image_file, calib_file, label_file, velodyne_file=None):
  image = cv2.imread(image_file)
  K, P2 = load_intrinsic(calib_file)
  image = draw_3d_box_on_image(image, label_file, P2)
  
  if velodyne_file is not None:
    range_list = [(-60, 60), (-100, 100), (-2., -2.), 0.1]
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    bev_image = points_filter.get_bev_image(velodyne_file)
    bev_image = draw_box_on_bev_image(bev_image, points_filter, label_file, calib_file)
    rows, cols = bev_image.shape[0], bev_image.shape[1]
    bev_image = bev_image[int(0.30*rows) : int(0.5*rows), int(0.20*cols): int(0.80*cols)]
    width = int(bev_image.shape[1])
    height = int(image.shape[0] * bev_image.shape[1] / image.shape[1])
    image = cv2.resize(image, (width, height))
    image = np.vstack([image, bev_image])
  return image

def pred_visual_tool(kitti_root, checkpoints_name, output_path, split="test"):
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


def kitti_visual_tool(kitti_root):
  if not os.path.exists(kitti_root):
    raise ValueError("kitti_root Not Found")

  image_path = os.path.join(kitti_root, "training/image_2")
  velodyne_path = os.path.join(kitti_root, "training/velodyne")
  calib_path = os.path.join(kitti_root, "training/calib")
  label_path = os.path.join(kitti_root, "training/label_2")
  image_ids = []
  for image_file in os.listdir(image_path):
    image_ids.append(image_file.split(".")[0])
  for i in range(len(image_ids)):
    image_2_file = os.path.join(image_path, str(image_ids[i]) + ".png")
    velodyne_file = os.path.join(velodyne_path, str(image_ids[i]) + ".bin")
    calib_file = os.path.join(calib_path, str(image_ids[i]) + ".txt")
    label_2_file = os.path.join(label_path, str(image_ids[i]) + ".txt")

    image = kitti_visual_tool_api(image_2_file, calib_file, label_2_file, velodyne_file)
    cv2.imshow("Image", image)
    cv2.waitKey(300)

def main():
  parser = argparse.ArgumentParser(description="Dataset in KITTI format Checking ...")
  parser.add_argument("--kitti_root", type=str,
                      default="",
                      help="Path to KITTI Dataset root")
  args = parser.parse_args()
  kitti_visual_tool(args.kitti_root)


if __name__ == "__main__":
  main()
