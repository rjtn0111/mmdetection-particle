# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .ade20k import ADE20KSegDataset


@DATASETS.register_module()
class CocoSegDataset(ADE20KSegDataset):
    """COCO dataset.

    In segmentation map annotation for COCO. The ``img_suffix`` is fixed to
    '.jpg',  and ``seg_map_suffix`` is fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood'
            ),
        palette=[
            (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
                 (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
                 (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
                 (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
                 (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
                 (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
                 (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
                 (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
                 (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
                 (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
                 (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
                 (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
                 (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
                 (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
                 (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
                 (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
                 (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
                 (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
                 (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
                 (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
                 (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
                 (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
                 (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
                 (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
                 (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
                 (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
                 (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
                 (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
                 (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
                 (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
                 (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
                 (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
                 (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
                 (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
                 (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
                 (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
                 (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
                 (102, 255, 0), (92, 0, 255), (107, 255, 200), (58, 41, 149),
                 (183, 121, 142), (255, 73, 97), (107, 142, 35),
                 (190, 153, 153), (146, 139, 141), (70, 130, 180),
                 (134, 199, 156), (209, 226, 140), (96, 36, 108), (96, 96, 96),
                 (64, 170, 64), (152, 251, 152), (208, 229, 228),
                 (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
                 (102, 102, 156), (250, 141, 255)
                 ])
