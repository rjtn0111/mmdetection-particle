_base_ = [
	'../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/alpha-rt_instance.py',
    '../_base_/schedules/schedule_50.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
	    frozen_stages=1,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50',
		              _delete_=True)))

# runtime setting
vis_backends = [
	dict(type='LocalVisBackend'),
	dict(type="WandbVisBackend",
      init_kwargs={'project': 'mmdetection-particle', 'name': 'mask-rcnn_r50_fpn_50_alpha-rt_pretrained'}), # add project name
    ]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')