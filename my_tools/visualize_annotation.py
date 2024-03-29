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


def visiualize_annotation(img_id: int, path=None, img_dir=None, output_dir=None, coco=None):
    print(f"- add segmentaion mask to original image")
    if coco == None:
        coco = COCO(path)

    img_data = coco.imgs[img_id]
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img_data['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    img = np.array(Image.open(img_dir + "/" + img_data['file_name']))
    plt.figure(figsize=(img_data["width"]/100, img_data["height"]/100), dpi=100)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(img)
    coco.showAnns(anns)  # ploting to current fig and ax 

    buf = io.BytesIO()
    plt.savefig(buf, format="png")  # save fig to buffer in order to save PIL image
    plt.close()
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize((img_data["width"], img_data["height"]))

    img_drawer = ImageDraw.Draw(img)
    for i, a in enumerate(anns):
        x, y, w, h = a["bbox"]
        color = PALETTE[a["category_id"]]
        img_drawer.rectangle([(x,y),(x+w, y+h)], outline=tuple(color), width=2)
        
    # Extract the file name from img_data['file_name']
    file_name = os.path.basename(img_data['file_name'])
    # Construct the output path by joining the 'outputs' directory and the file name
    output_path = os.path.join(output_dir, file_name) # 保存場所
    img.save(output_path)


def convert_all_images(path, img_dir, output_dir):
    coco = COCO(path)
    
    # すべての画像IDを取得
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        visiualize_annotation(img_id, path, img_dir, output_dir)
        

# path = '../data/coco/annotations/instances_val2017.json'
path = '../data/test/annotations/test_annotation.json'
img_dir = '../data/test/imgs/'
output_dir = "./outputs/vis_annotation/"

convert_all_images(path, img_dir, output_dir)