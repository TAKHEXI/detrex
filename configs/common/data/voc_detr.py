from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    # MetadataCatalog
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detrex.data import DetrDatasetMapper


register_coco_instances("coco_vehicle_train", {}, "/home/xyyu/datasets/coco/annotations/instances_train2014(vehicle-base).json", "/home/xyyu/datasets/coco/train2014")
register_coco_instances("coco_vehicle_val", {}, "/home/xyyu/datasets/coco/annotations/instances_val2014(vehicle-base).json", "/home/xyyu/datasets/coco/val2014")
# register_coco_instances("coco_vehicle_test", {}, "/home/xyyu/datasets/coco/annotations/image_info_test2014.json", "/home/xyyu/datasets/coco/test2014")
# MetadataCatalog.get("coco_vehicle_train").thing_classes=["bicyle", "motorcycle", "bus", "truck"]
# MetadataCatalog.get("coco_vehicle_val").thing_classes=["bicyle", "motorcycle", "bus", "truck"]

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_vehicle_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_vehicle_val", filter_empty=False),
    # dataset=L(get_detection_dataset_dicts)(names="coco_vehicle_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
