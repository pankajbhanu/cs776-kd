from torch.optim.sgd import SGD
from kd.engine.runner import Runner
from kd.models.cvops.nms import NMSop
from kd.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from kd.models.detectors import CrossKDRetinaNet
from kd.models.backbones import ResNet
from kd.models.necks import FPN
from kd.models.dense_heads import RetinaHead
from kd.models.task_modules.prior_generators import AnchorGenerator
from kd.models.task_modules.coders import DeltaXYWHBBoxCoder
from kd.models.losses import FocalLoss, L1Loss, GIoULoss, KDQualityFocalLoss
from kd.models.task_modules.assigners import MaxIoUAssigner
from kd.engine.optim import OptimWrapper
from kd.engine.hooks import CheckpointHook, LoggerHook
from kd.models.task_modules.samplers.pseudo_sampler import PseudoSampler

from kd.engine.dataset import CocoDataset
from torch.utils.data import DataLoader

from .test_dataset import train_dataset



def main():

    """

     dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

    dataset = DATASETS.build(dataset_cfg)



     data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader






        def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):


    train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
    """

    train_dataset = train_dataset


    # sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    sampler = None                  # TODO: should use torch sampler or get from engine

    # train dataloader
    train_dataloader = DataLoader(
        dataset = train_dataset,
        sampler = sampler,
        # batch_sampler=None,
    )


    # DetDataPreprocessor forward method expects 
    # data (dict): Data sampled from dataloader.
    # should see where is it being called from.
    


    # Define the training configuration inline.
    # This dictionary replicates what you would normally place in a config file.
    teacher_data_preprocessor = DetDataPreprocessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )
    teacher_resnet = ResNet(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        pretrained="torchvision://resnet50",
    )
    teacher_neck = FPN(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    )
    teacher_anchor_generator = AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128]
    )
    teacher_bbox_coder = DeltaXYWHBBoxCoder(
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]
    )
    teacher_loss_cls = FocalLoss(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0
    )
    teacher_loss_bbox = L1Loss(loss_weight=1.0)
    teacher_maxiou_assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1
    )
    teacher_sampler = PseudoSampler()
    teacher_nms = NMSop(iou_threshold=0.5)
    teacher_bbox_head = RetinaHead(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
        anchor_generator=teacher_anchor_generator,
        bbox_coder=teacher_bbox_coder,
        loss_cls=teacher_loss_cls,
        loss_bbox=teacher_loss_bbox,
    )
    student_resnet = ResNet(
        depth=18,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        pretrained="torchvision://resnet18"
    )
    student_neck = FPN(
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
        start_level=1,
        add_extra_convs="on_output",
    )
    student_anchor_generator = AnchorGenerator(
        strides=[8, 16, 32, 64, 128],
        ratios=[0.5, 1.0, 2.0],
        octave_base_scale=4,
        scales_per_octave=3,
    )
    student_bbox_coder = DeltaXYWHBBoxCoder(
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0]
    )
    student_loss_cls = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
    student_loss_bbox = L1Loss(loss_weight=1.0)
    student_bbox_head = RetinaHead(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
        anchor_generator=student_anchor_generator,
        bbox_coder=student_bbox_coder,
        loss_cls=student_loss_cls,
        loss_bbox=student_loss_bbox,
    )
    teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    detector = CrossKDRetinaNet(
        backbone=student_resnet,
        neck=student_neck,
        bbox_head=student_bbox_head,
        teacher=teacher,
        teacher_ckpt=teacher_ckpt
    )
    loss_cls_kd = KDQualityFocalLoss(beta=1, loss_weight=1.0)
    loss_reg_kd = GIoULoss(loss_weight=1.0)
    reused_teacher_head_index = 3
    student_assigner = MaxIoUAssigner(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
    syudent_sampler = PseudoSampler()
    student_nms = NMSop(iou_threshold=0.5)
    optim_wrapper = OptimWrapper(
        optimizer=SGD(lr=0.01, momentum=0.9, weight_decay=0.0001)
    )
    kd_cfg=dict(
                loss_cls_kd=loss_cls_kd,
                loss_reg_kd=loss_reg_kd,
                reused_teacher_head_idx=reused_teacher_head_index,
            )
    checkpoint_hook = CheckpointHook()
    logger_hook = LoggerHook()
    cfg = dict(
        model=dict(
            type="CrossKDRetinaNet",
            data_preprocessor=dict(
                type="DetDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_size_divisor=32,
            ),
            teacher_config="configs/retinanet/retinanet_r50_fpn_1x_coco.py",
            teacher_ckpt="https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
            backbone=dict(
                type="ResNet",
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=True),
                norm_eval=True,
                style="pytorch",
                init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
            ),
            neck=dict(
                type="FPN",
                in_channels=[64, 128, 256, 512],
                out_channels=256,
                start_level=1,
                add_extra_convs="on_output",
                num_outs=5,
            ),
            bbox_head=dict(
                type="RetinaHead",
                num_classes=80,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                anchor_generator=dict(
                    type="AnchorGenerator",
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128],
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[1.0, 1.0, 1.0, 1.0],
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            kd_cfg=dict(
                loss_cls_kd=dict(type="KDQualityFocalLoss", beta=1, loss_weight=1.0),
                loss_reg_kd=dict(type="GIoULoss", loss_weight=1.0),
                reused_teacher_head_idx=3,
            ),
            train_cfg=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(type="PseudoSampler"),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5),
                max_per_img=100,
            ),
        ),
        work_dir="./work_dirs/crosskd_r18_retinanet_r50_fpn_1x_coco",
        train_dataloader=dict(
            dataset=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/instances_train2017.json",
                img_prefix="data/coco/train2017/",
            ),
            sampler=dict(type="DefaultSampler", shuffle=True),
            batch_size=2,
            num_workers=4,
            collate_fn=dict(type="pseudo_collate"),
        ),
        train_cfg=dict(by_epoch=True, max_epochs=12, val_interval=1),
        optim_wrapper = OptimWrapper(
            optimizer=SGD(lr=0.01, momentum=0.9, weight_decay=0.0001),
            accumulative_counts=1,
            clip_grad=dict(max_norm=35, norm_type=2),
        ),
        # optim_wrapper=dict(
        #     type="OptimWrapper",
        #     optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001),
        # ),
        auto_scale_lr=dict(enable=True, base_batch_size=16),
        default_hooks=dict(
            checkpoint=CheckpointHook(),
            logger=LoggerHook(),
        ),
        launcher="none",
    )

    # Build the runner from the configuration dictionary.
    runner = Runner(detector)
    # Start training.
    runner.train()


if __name__ == "__main__":
    main()
