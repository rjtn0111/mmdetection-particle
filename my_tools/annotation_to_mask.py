# inference: https://zenn.dev/inaturam/articles/2fe1e679b74e4b
import io
import json
import textwrap
import os

from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

PALETTE = [
    [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
    [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
    [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
    [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
    [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
    [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
    [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
    [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
    [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
    [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
    [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
    [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
    [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
    [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
    [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
    [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
    [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
    [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
    [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
    [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
    [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
    [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
    [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
    [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
    [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
    [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
    [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
    [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
    [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
    [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
    [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
    [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
    [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
    [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
    [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
    [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
    [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
    [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
    [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
    [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
    [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
    [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
    [64, 192, 96], [64, 160, 64], [64, 64, 0],
    ]  # COCO-Stuff dataset patlette

def annotation_to_mask(img_id: int, path=None, output_dir=None, coco=None):
    print(f"- make segmentaion mask from annotaion")

    if coco == None:
        coco = COCO(path)
    # img_ids = coco.getImgIds()
    img_data = coco.imgs[img_id]
    print(f"load image from: {img_data['file_name']}")

    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img_data['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    
    mask = coco.annToMask(anns[0])  # sampling annotation mask from anno_id 0 as a canvas.
    mask = np.stack([mask, mask, mask], axis=2) * 0  # dim = 2 -> [h,w,3]

    for i, a in enumerate(anns):
        color = PALETTE[a["category_id"]]
        one_mask = np.where(coco.annToMask(a) > 0, 1, 0)
        one_mask = np.stack([one_mask * color[0], one_mask * color[1], one_mask * color[2]], axis=2)
        mask = np.where(one_mask > 0, one_mask, mask)  # overlap non-zero mask on previous mask
    
    # Extract the file name from img_data['file_name']
    file_name = os.path.basename(img_data['file_name'])
    # Construct the output path by joining the 'outputs' directory and the file name
    output_path = os.path.join(output_dir, file_name) # 保存場所
    
    mask = Image.fromarray(np.uint8(mask))
    mask.save(output_path)


def convert_all_images(path, output_dir):
    coco = COCO(path)
    
    # すべての画像IDを取得
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        annotation_to_mask(img_id, path, output_dir)


# path = '../data/alpha-rt/annotation/val/val.json'
# path = '../data/coco/annotations/instances_val2017.json'
path = '../data/test/annotations/test_annotation.json'
output_dir = "./outputs/mask_from_annotation/"

convert_all_images(path, output_dir)
# 0 image only
# annotation_to_mask(1, path)