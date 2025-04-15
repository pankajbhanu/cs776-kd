"""
https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html
"""


from kd.models.detdataset.coco import CocoDataset
from kd.cv.transforms import LoadImageFromFile, LoadAnnotations, Resize, RandomFlip
from kd.models.detdataset.transforms import PackDetInputs

import torch
from torch.utils.data import DataLoader
from kd.engine.dataset import DefaultSampler, pseudo_collate

from kd.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from kd.models.backbones import ResNet
from kd.models.necks import FPN
from kd.models.task_modules.prior_generators import AnchorGenerator
from kd.models.task_modules.coders import DeltaXYWHBBoxCoder
from kd.models.losses import FocalLoss, L1Loss, GIoULoss, KDQualityFocalLoss
from kd.models.dense_heads import RetinaHead
"""
---------------1. dataset
"""
data_root = 'dataset/sample_mini'
file_client_args = dict(backend='disk')

train_pipeline = [
    LoadImageFromFile(file_client_args=file_client_args),
    LoadAnnotations(with_bbox=True, ),
    Resize(scale=(1333, 800), keep_ratio=True),
    RandomFlip(prob=0.5),
    PackDetInputs()
]


train_dataset = CocoDataset(
    ann_file='sample_instances_train2017.json',
    data_root = data_root,
    data_prefix=dict(img='sample_train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

# print(train_dataset.metainfo)
# print(train_dataset.get_data_info(0))


"""
--------------2. data loader
"""

sampler = DefaultSampler(train_dataset, shuffle=True)

train_dataloader = DataLoader(
    dataset=train_dataset,
    sampler=sampler,
    collate_fn=pseudo_collate,
    batch_size=7,
)

# print(train_dataloader)

# # Display image and label.
# inputs, data_samples = next(iter(train_dataloader))

# # print(f"Feature batch shape: {inputs.size()}")
# # print(f"Labels batch shape: {data_samples.size()}")
# # img = train_features[0].squeeze()
# # label = train_labels[0]


batch = next(iter(train_dataloader))
inputs = batch['inputs']
data_samples = batch['data_samples']

# print(batch)
# print(inputs[0])
# print(data_samples[0])
# print(type(batch))
# print(type(inputs))
# print(len(inputs))
# print(type(data_samples))


"""
-------------------3. Det Preprocessor
"""


# from mmdet.models import DetDataPreprocessor



data_preprocessor = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )

# print(data_preprocessor)
# print(type(data_preprocessor))



data_pre = data_preprocessor(batch)
# print(data_pre)
# print(type(data_pre))                   # DIct

inputs = data_pre['inputs']
data_samples = data_pre['data_samples']

# print(type(inputs))                     # Tensor
# print(type(data_samples))               # List

# print(inputs[0])
# print(data_samples[0])

print(inputs.shape)                 # (5, 3, 1216, 1248])   B N H W




"""
4. backbone (teacher)

"""


teacher_resnet = ResNet(
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    pretrained="torchvision://resnet50",
)

# print(teacher_resnet)

x = teacher_resnet(inputs)
# print(x.shape)
# print(x)
# print(type(x))



"""
5. neck (teacher)

"""



teacher_neck = FPN(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=1,
    add_extra_convs='on_input',
    num_outs=5
)

# print(dir(teacher_neck))
x = teacher_neck(x)


# print(x)
# print(len(x))
# print(type(x))

a,b,c,d,e = x
# print(a,b,c,d,e)



# conv0 = teacher_neck.fpn_convs[0]
# print(conv0)
# print(type(conv0))

# conv0_out = conv0(x)
# print(conv0(x))
# print(x)
# print(x.shape)

# print(teacher_neck.fpn_convs)
# print(teacher_neck.lateral_convs)
# print(teacher_neck.num_ins)
# print(teacher_neck.num_outs)
# print(teacher_neck.in_channels)
# print(teacher_neck.out_channels)
# print(teacher_neck.upsample_cfg)




"""

6. anchor generation and bbox coder
"""



teacher_anchor_generator = AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128]
    )

print(teacher_anchor_generator)

teacher_bbox_coder = DeltaXYWHBBoxCoder(
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0]
)

print(teacher_bbox_coder)
# print(teacher_bbox_coder(a))



"""
7. loss
"""



teacher_loss_cls = FocalLoss(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0
    )
teacher_loss_bbox = L1Loss(loss_weight=1.0)

print(teacher_loss_bbox)
print(teacher_loss_cls)


"""
8.head




bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
"""


teacher_bbox_head = RetinaHead(
    anchor_generator=teacher_anchor_generator,
    bbox_coder=teacher_bbox_coder,
    loss_cls=teacher_loss_cls,
    loss_bbox=teacher_loss_bbox,
    num_classes=80,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
)
print(teacher_bbox_head)
# print(teacher_bbox_head(a))


# print(dir(teacher_bbox_head))

print(teacher_bbox_head.loss_cls)
print(teacher_bbox_head.loss_bbox)

# print(data_samples[0])







"""

student
"""




"""
AttributeError: 'CocoDataset' object has no attribute 'file_client'
"""


"""
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
"""

"""
---------------1. dataset
"""
data_root = 'dataset/sample_mini/'
file_client_args = dict(backend='disk')

train_pipeline = [
    LoadImageFromFile(file_client_args=file_client_args),
    LoadAnnotations(with_bbox=True, ),
    Resize(scale=(1333, 800), keep_ratio=True),
    RandomFlip(prob=0.5),
    PackDetInputs()
]


train_dataset = CocoDataset(
    ann_file='sample_instances_train2017.json',
    data_root = data_root,
    data_prefix=dict(img='sample_train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

# print(train_dataset.metainfo)
# print(train_dataset.get_data_info(0))


"""
--------------2. data loader
"""



sampler = DefaultSampler(train_dataset, shuffle=True)



train_dataloader = DataLoader(
    dataset=train_dataset,
    sampler=sampler,
    collate_fn=pseudo_collate,
    batch_size=7,
)

# print(train_dataloader)

# # Display image and label.
# inputs, data_samples = next(iter(train_dataloader))

# # print(f"Feature batch shape: {inputs.size()}")
# # print(f"Labels batch shape: {data_samples.size()}")
# # img = train_features[0].squeeze()
# # label = train_labels[0]


batch = next(iter(train_dataloader))
inputs = batch['inputs']
data_samples = batch['data_samples']

# print(batch)
# print(inputs[0])
# print(data_samples[0])
# print(type(batch))
# print(type(inputs))
# print(len(inputs))
# print(type(data_samples))




"""
-------------------3. Det Preprocessor
"""



data_preprocessor = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )

# print(data_preprocessor)
# print(type(data_preprocessor))



data_pre = data_preprocessor(batch)
# print(data_pre)
# print(type(data_pre))                   # DIct

inputs = data_pre['inputs']
data_samples = data_pre['data_samples']

# print(type(inputs))                     # Tensor
# print(type(data_samples))               # List

# print(inputs[0])
# print(data_samples[0])

print(inputs.shape)                 # (5, 3, 1216, 1248])   B N H W




"""
4. backbone (teacher)

"""

student_resnet = ResNet(
    depth=18,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    pretrained="torchvision://resnet18",
)

# teacher_resnet = ResNet(
#     depth=50,
#     num_stages=4,
#     out_indices=(0, 1, 2, 3),
#     frozen_stages=1,
#     pretrained="torchvision://resnet50",
# )

# print(teacher_resnet)

x = student_resnet(inputs)
# print(x.shape)
# print(x)
# print(type(x))



"""
5. neck (teacher)

"""


student_neck = FPN(
    in_channels=[64, 128, 256, 512],
    out_channels=256,
    start_level=1,
    add_extra_convs='on_output',
    num_outs=5
)

# print(dir(teacher_neck))
x = student_neck(x)


# print(x)
# print(len(x))
# print(type(x))

a,b,c,d,e = x
# print(a,b,c,d,e)



# conv0 = teacher_neck.fpn_convs[0]
# print(conv0)
# print(type(conv0))

# conv0_out = conv0(x)
# print(conv0(x))
# print(x)
# print(x.shape)

# print(teacher_neck.fpn_convs)
# print(teacher_neck.lateral_convs)
# print(teacher_neck.num_ins)
# print(teacher_neck.num_outs)
# print(teacher_neck.in_channels)
# print(teacher_neck.out_channels)
# print(teacher_neck.upsample_cfg)




"""

6. anchor generation and bbox coder
"""


student_anchor_generator = AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128]
    )

print(student_anchor_generator)

student_bbox_coder = DeltaXYWHBBoxCoder(
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0]
)

print(student_bbox_coder)
# print(teacher_bbox_coder(a))



"""
7. loss
"""


student_loss_cls = FocalLoss(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0
    )
student_loss_bbox = L1Loss(loss_weight=1.0)

print(student_loss_cls)
print(student_loss_bbox)


"""
8.head


bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
"""

student_bbox_head = RetinaHead(
    anchor_generator=teacher_anchor_generator,
    bbox_coder=teacher_bbox_coder,
    loss_cls=teacher_loss_cls,
    loss_bbox=teacher_loss_bbox,
    num_classes=80,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
)
print(student_bbox_head)
# print(teacher_bbox_head(a))

# print(data_samples[0])

loss_cls_kd = KDQualityFocalLoss(beta=1, loss_weight=1.0)
loss_reg_kd = GIoULoss(loss_weight=1.0)
reused_teacher_head_index = 3
student_assigner = MaxIoUAssigner(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
syudent_sampler = PseudoSampler()
student_nms = NMSop(iou_threshold=0.5)
optim_wrapper = OptimWrapper(
    optimizer=SGD(lr=0.01, momentum=0.9, weight_decay=0.0001)
)
checkpoint_hook = CheckpointHook(interval=12)
logger_hook = LoggerHook
