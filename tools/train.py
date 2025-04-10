from kd.config.common.optim import SGD
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


def main():
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
    checkpoint_hook = CheckpointHook(interval=12)
    logger_hook = LoggerHook
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
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001),
        ),
        auto_scale_lr=dict(enable=True, base_batch_size=16),
        default_hooks=dict(
            checkpoint=dict(type="CheckpointHook", interval=12),
            logger=dict(type="LoggerHook"),
        ),
        launcher="none",
    )

    # Build the runner from the configuration dictionary.
    runner = Runner(detector)
    # Start training.
    runner.train()


if __name__ == "__main__":
    main()
