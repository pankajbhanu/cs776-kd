import sys
import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog

if __name__ == "__main__":
    setup_logger()

    filename = sys.argv[1]
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml")) # Or your modified config file
    cfg.merge_from_file("retinanet.yaml")
     # For COCO format:
    register_coco_instances("my_dataset_train", {}, "dataset/instances_minitrain2017.json", "dataset/coco_minitrain2017_25k/images")
    register_coco_instances("my_dataset_val", {}, "dataset/instances_minitrain2017.json", "dataset/coco_minitrain2017_25k/images")
    # Get metadata (optional but recommended)
    my_dataset_metadata_train = MetadataCatalog.get("my_dataset_train")
    my_dataset_metadata_val = MetadataCatalog.get("my_dataset_val")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    # # List all registered datasets
    # print(DatasetCatalog.get("my_dataset_train")[0])
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml") # Load pretrained weights
    cfg.MODEL.RETINANET.NUM_CLASSES = 80 # Or the number of classes in your dataset
    cfg.OUTPUT_DIR = "./output"

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = (20000, 25000)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for inference

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    torch.cuda.empty_cache()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False) # Set resume=True to continue training
    trainer.train()