Create Dataset
==============

The data pipeline uses standard COCO format via ``torchvision.datasets.CocoDetection``.

Dataset Format
--------------

The pipeline expects COCO format annotations:

.. code-block:: text

    data/coco/
    |-- train2017/
    |   |-- 000000000001.jpg
    |   |-- ...
    |-- val2017/
    |   |-- 000000000001.jpg
    |   |-- ...
    |-- annotations/
        |-- instances_train2017.json
        |-- instances_val2017.json

DataModule
----------

The ``YOLODataModule`` handles all data loading:

.. code-block:: python

    from yolo.data.datamodule import YOLODataModule

    data = YOLODataModule(
        root="data/coco",
        train_images="train2017",
        val_images="val2017",
        train_ann="annotations/instances_train2017.json",
        val_ann="annotations/instances_val2017.json",
        batch_size=16,
        image_size=[640, 640],
        num_workers=8,
    )

Configuration
~~~~~~~~~~~~~

Data configuration in ``yolo/config/experiment/default.yaml``:

.. code-block:: yaml

    data:
      root: data/coco
      train_images: train2017
      val_images: val2017
      train_ann: annotations/instances_train2017.json
      val_ann: annotations/instances_val2017.json
      batch_size: 16
      num_workers: 8
      image_size: [640, 640]
      # Augmentation
      mosaic_prob: 1.0
      mixup_prob: 0.15
      flip_lr: 0.5

Custom Dataset
--------------

For custom datasets, create COCO-format annotations and configure the paths:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
        --data.root=data/my_dataset \
        --data.train_images=images/train \
        --data.val_images=images/val \
        --data.train_ann=annotations/train.json \
        --data.val_ann=annotations/val.json \
        --model.num_classes=10

Dataloader Return Type
~~~~~~~~~~~~~~~~~~~~~~

For each batch, the dataloader returns:

- **images**: Tensor of shape ``[batch_size, 3, height, width]``
- **targets**: List of dictionaries with COCO-format annotations

Augmentations
-------------

The following augmentations are applied during training:

- **Mosaic**: Combines 4 images into one
- **MixUp**: Blends two images together
- **HSV**: Random hue, saturation, value adjustments
- **Flip**: Horizontal flip
- **Resize**: Letterbox resize to target size
