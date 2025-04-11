from .coco import CocoDataset

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataset = CocoDataset(
    ann_file='annotations/instances_train2017.json',
    data_root = data_root,
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

print(train_dataset)