Train & Validation
==================

Training Model
--------------

Training is handled by PyTorch Lightning. Use the CLI to start training:

.. code-block:: bash

    python -m yolo.cli fit --config yolo/config/experiment/default.yaml

Or use the modules directly in Python:

.. code-block:: python

    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from yolo.training.module import YOLOModule
    from yolo.data.datamodule import YOLODataModule

    # Create model and data module
    model = YOLOModule(
        model_config="v9-c",
        num_classes=80,
        learning_rate=0.01,
    )
    data = YOLODataModule(
        root="data/coco",
        batch_size=16,
        image_size=[640, 640],
    )

    # Configure callbacks
    callbacks = [
        ModelCheckpoint(monitor="val/mAP", mode="max", save_top_k=3),
        EarlyStopping(monitor="val/mAP", mode="max", patience=50),
    ]

    # Create trainer and fit
    trainer = Trainer(
        max_epochs=500,
        accelerator="auto",
        precision="16-mixed",
        callbacks=callbacks,
    )
    trainer.fit(model, data)


Training Diagram
~~~~~~~~~~~~~~~~

The following diagram illustrates the training process:

.. mermaid::

  flowchart LR
    subgraph Lightning["Trainer.fit()"]
      subgraph TE["training_step"]
        forward-->loss
        loss-->backward
      end
      subgraph VE["validation_step"]
        VF[forward]-->metrics
      end
    end
    TE-->VE
    VE-->TE


Validation Model
----------------

To validate a trained model:

.. code-block:: bash

    python -m yolo.cli validate --config yolo/config/experiment/default.yaml \
        --ckpt_path=runs/best.ckpt

Or in Python:

.. code-block:: python

    trainer = Trainer()
    trainer.validate(model, data, ckpt_path="runs/best.ckpt")

Metrics
~~~~~~~

The following metrics are computed during validation:

- ``val/mAP`` - mAP @ IoU=0.50:0.95 (COCO primary metric)
- ``val/mAP50`` - mAP @ IoU=0.50
- ``val/mAP75`` - mAP @ IoU=0.75
- ``val/mAP_small`` - mAP for small objects (< 32x32 px)
- ``val/mAP_medium`` - mAP for medium objects (32-96 px)
- ``val/mAP_large`` - mAP for large objects (> 96x96 px)
