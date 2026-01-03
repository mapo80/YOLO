All In 1
========

This guide covers training, validation, and inference using PyTorch Lightning CLI.

Train Model
-----------

To train the model:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/default.yaml

Common Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All configuration is done via YAML files in ``yolo/config/experiment/``. You can override any parameter from the command line:

**Model parameters:**

- ``--model.model_config``: Model architecture (v9-c, v9-s, v9-m, etc.)
- ``--model.num_classes``: Number of classes (default: 80)
- ``--model.learning_rate``: Learning rate (default: 0.01)

**Data parameters:**

- ``--data.batch_size``: Batch size (default: 16)
- ``--data.image_size``: Input image size (default: [640, 640])
- ``--data.num_workers``: Number of dataloader workers (default: 8)

**Trainer parameters:**

- ``--trainer.max_epochs``: Total training epochs (default: 500)
- ``--trainer.accelerator``: Device type (auto, gpu, cpu, mps)
- ``--trainer.devices``: Number of devices (auto, 1, 2, etc.)
- ``--trainer.precision``: Training precision (16-mixed, 32, bf16-mixed)

Examples
~~~~~~~~

Train with custom batch size and learning rate:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
        --data.batch_size=32 \
        --model.learning_rate=0.001 \
        --trainer.max_epochs=100

Quick debug run:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/debug.yaml


Multi-GPU Training
~~~~~~~~~~~~~~~~~~

PyTorch Lightning handles multi-GPU training automatically:

.. code-block:: bash

    # Single GPU
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
        --trainer.devices=1

    # Multiple GPUs with DDP
    python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
        --trainer.devices=2 \
        --trainer.strategy=ddp


Custom Dataset
~~~~~~~~~~~~~~

The pipeline uses standard **COCO format** via ``torchvision.datasets.CocoDetection``.

Expected directory structure:

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

To train on a custom dataset:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
        --data.root=data/my_dataset \
        --data.train_images=images/train \
        --data.val_images=images/val \
        --data.train_ann=annotations/train.json \
        --data.val_ann=annotations/val.json \
        --model.num_classes=10


Validation
----------

To validate a trained model:

.. code-block:: bash

    python -m yolo.cli validate --config yolo/config/experiment/default.yaml \
        --ckpt_path=runs/best.ckpt

Metrics logged during validation:

- ``val/mAP``: mAP @ IoU=0.50:0.95 (COCO primary metric)
- ``val/mAP50``: mAP @ IoU=0.50
- ``val/mAP75``: mAP @ IoU=0.75
- ``val/mAP_small``: mAP for small objects
- ``val/mAP_medium``: mAP for medium objects
- ``val/mAP_large``: mAP for large objects


Inference
---------

To run inference on images:

.. code-block:: bash

    python examples/sample_inference.py --image path/to/image.jpg

    # With custom weights
    python examples/sample_inference.py --image path/to/image.jpg --weights weights/v9-c.pt

    # With custom model config
    python examples/sample_inference.py --image path/to/image.jpg --model v9-s
