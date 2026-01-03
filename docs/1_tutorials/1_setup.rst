Setup Config
============

Configuration is managed through YAML files and PyTorch Lightning CLI.

Configuration Files
-------------------

All training configuration is stored in ``yolo/config/experiment/``:

- ``default.yaml`` - Full training configuration
- ``debug.yaml`` - Quick debug configuration

Model architectures are defined in ``yolo/config/model/``:

- ``v9-c.yaml``, ``v9-s.yaml``, ``v9-m.yaml``, etc.

Configuration Structure
-----------------------

The experiment config has three main sections:

**Trainer** - PyTorch Lightning Trainer settings:

.. code-block:: yaml

    trainer:
      max_epochs: 500
      accelerator: auto
      devices: auto
      precision: 16-mixed

**Model** - YOLOModule settings:

.. code-block:: yaml

    model:
      model_config: v9-c
      num_classes: 80
      learning_rate: 0.01

**Data** - YOLODataModule settings:

.. code-block:: yaml

    data:
      root: data/coco
      batch_size: 16
      image_size: [640, 640]

CLI Override
------------

Any parameter can be overridden from the command line:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
        --model.learning_rate=0.001 \
        --data.batch_size=32 \
        --trainer.max_epochs=100

Programmatic Usage
------------------

You can also use the modules directly in Python:

.. code-block:: python

    from lightning import Trainer
    from yolo.training.module import YOLOModule
    from yolo.data.datamodule import YOLODataModule

    # Create model and data module
    model = YOLOModule(model_config="v9-c", num_classes=80)
    data = YOLODataModule(root="data/coco", batch_size=16)

    # Create trainer and fit
    trainer = Trainer(max_epochs=100, accelerator="auto")
    trainer.fit(model, data)
