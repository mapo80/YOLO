Build Model
===========

In YOLOv7, the prediction will be ``Anchor``, and in YOLOv9, it will predict ``Vector``. The converter will turn the bounding box to the vector.

The overall model flowchart is as follows:

.. mermaid::

    flowchart LR
    Input-->Model;
    Model--Class-->NMS;
    Model--Anc/Vec-->Converter;
    Converter--Box-->NMS;
    NMS-->Output;

Load Model
~~~~~~~~~~

Using ``create_model``, it will automatically create the YOLO model based on the architecture config.

.. code-block:: python

    from yolo import create_model

    # Create model from architecture name
    model = create_model("v9-c", num_classes=80)
    model = model.to(device)

    # Load pretrained weights
    model.load_state_dict(torch.load("weights/v9-c.pt"))

Model Architecture
~~~~~~~~~~~~~~~~~~

Model architectures are defined in YAML files under ``yolo/config/model/``:

- ``v9-c.yaml`` - YOLOv9-C (compact)
- ``v9-s.yaml`` - YOLOv9-S (small)
- ``v9-m.yaml`` - YOLOv9-M (medium)

The architecture uses a custom DSL (Domain Specific Language) to define layers:

.. code-block:: yaml

    model:
      backbone:
        - Conv:
            args: {out_channels: 64, kernel_size: 3, stride: 2}
        - RepNCSPELAN:
            args: {out_channels: 256}
            tags: B3

      neck:
        - SPPELAN:
            args: {out_channels: 512}
        - Concat:
            source: [-1, B3]

      head:
        - MultiheadDetection:
            source: [P3, P4, P5]
            output: True

Converter
~~~~~~~~~

The converter transforms model predictions to bounding boxes:

.. code-block:: python

    from yolo import create_converter

    converter = create_converter(
        model_name="v9-c",
        model=model,
        anchor_cfg=anchor_config,
        image_size=[640, 640],
        device=device
    )
