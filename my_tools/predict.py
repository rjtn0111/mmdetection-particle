# https://qiita.com/bellpond/items/33c15c8eb62f46a51aba
# https://github.com/open-mmlab/mmdetection/issues/10380

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

# モデルの設定ファイルと学習済モデルへのパスを指定する
config_file = '../my_configs/mask-rcnn/mask-rcnn_r50_fpn_200_alpha-rt.py'
checkpoint_file = '../work_dirs/mask-rcnn_r50_fpn_200_alpha-rt/epoch_110.pth'

# モデルの初期化
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# テストする画像を指定
img = '../data/alpha-rt/imgs/val/00007500.png'
img = mmcv.imread(img)

# テスト
result = inference_detector(model, img)

# init the visualizer(execute this block only once)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta


# show the results
visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    wait_time=0,
    out_file='outputs/predict.png' # optionally, write to output file
)
# visualizer.show()