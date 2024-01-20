import json
import collections as cl
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
import cv2
import glob
import sys
import os

### https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f

def info():
    tmp = cl.OrderedDict()
    tmp["description"] = "Test"
    tmp["url"] = "https://test"
    tmp["version"] = "0.01"
    tmp["year"] = 2020
    tmp["contributor"] = "salt22g"
    tmp["data_created"] = "2020/12/20"
    return tmp

def licenses():
    tmp = cl.OrderedDict()
    tmp["id"] = 1
    tmp["url"] = "dummy_words"
    tmp["name"] = "salt22g"
    return tmp

def images(mask_path):
    tmps = []
    files = glob.glob(mask_path + "/*.png")
    files.sort()

    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        height, width = img.shape[:3]

        tmp = cl.OrderedDict()
        tmp["license"] = 1
        tmp["id"] = i
        tmp["file_name"] = os.path.basename(file)
        tmp["width"] = width
        tmp["height"] = height
        tmp["date_captured"] = ""
        tmp["coco_url"] = "dummy_words"
        tmp["flickr_url"] = "dummy_words"
        tmps.append(tmp)
    return tmps


def annotations(mask_path):
    tmps = []

    files = glob.glob(mask_path + "/*.png")
    files.sort()
    
    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        tmp = cl.OrderedDict()
        segmentation_list = []
        contours = measure.find_contours(img, 0.5)

        for contour in contours:
            for a in contour:
                segmentation_list.append(a[1])
                segmentation_list.append(a[0])

        mask = np.array(img)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []

        for j in range(num_objs):
            pos = np.where(masks[j])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        tmp_segmentation = cl.OrderedDict()

        tmp["segmentation"] = [segmentation_list]
        tmp["id"] = str(i)
        tmp["image_id"] = i
        tmp["category_id"] = 1
        tmp["area"] = float(boxes[0][3] - boxes[0][1]) * float(boxes[0][2] - boxes[0][0])
        tmp["iscrowd"] = 0
        tmp["bbox"] =  [float(boxes[0][0]), float(boxes[0][1]), float(boxes[0][3] - boxes[0][1]), float(boxes[0][2] - boxes[0][0])]
        tmps.append(tmp)
    return tmps

def categories():
    tmps = []
    sup = ["target"]
    cat = ["target"]
    for i in range(len(sup)):
        tmp = cl.OrderedDict()
        tmp["id"] = i+1
        tmp["name"] = cat[i]
        tmp["supercategory"] = sup[i]
        tmps.append(tmp)
    return tmps

def main(mask_path, json_name):
    query_list = ["info", "licenses", "images", "annotations", "categories", "segment_info"]
    js = cl.OrderedDict()
    for i in range(len(query_list)):
        tmp = ""
        # Info
        if query_list[i] == "info":
            tmp = info()

        # licenses
        elif query_list[i] == "licenses":
            tmp = licenses()

        elif query_list[i] == "images":
            tmp = images(mask_path)

        elif query_list[i] == "annotations":
            tmp = annotations(mask_path)

        elif query_list[i] == "categories":
            tmp = categories()

        # save it
        js[query_list[i]] = tmp

    # write
    fw = open(json_name,'w')
    json.dump(js,fw,indent=2)

args = sys.argv
mask_path = args[1]
#mask_path =  ""
json_name = args[2]
#json_name = "person_sample.json"

if __name__=='__main__':
    main(mask_path, json_name)