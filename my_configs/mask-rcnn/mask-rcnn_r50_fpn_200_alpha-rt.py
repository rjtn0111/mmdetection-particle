_base_ = [
	'../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/alpha-rt_instance.py',
    '../_base_/schedules/schedule_200.py',
    '../_base_/default_runtime.py'
]


# runtime setting
vis_backends = [
	dict(type='LocalVisBackend'),
	dict(type="WandbVisBackend",
      init_kwargs={'project': 'mmdetection-particle', 'name': 'mask-rcnn_r50_fpn_300_alpha-rt'}), # add project name
    ]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')