"""
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
"""

from torch.utils.data import DataLoader
from tools.a import train_dataset

from mmengine.dataset import DefaultSampler, default_collate



sampler = DefaultSampler(train_dataset, shuffle=True)



train_dataloader = DataLoader(
    dataset=train_dataset,
    sampler=sampler,
    collate_fn=default_collate,
    batch_size=5,
)

print(train_dataloader)
