Inference
=========

Running Inference
-----------------

Use the sample inference script to run detection on images:

.. code-block:: bash

    python examples/sample_inference.py --image path/to/image.jpg

    # With custom weights
    python examples/sample_inference.py --image path/to/image.jpg --weights weights/v9-c.pt

    # With different model architecture
    python examples/sample_inference.py --image path/to/image.jpg --model v9-s


Python API
----------

For programmatic inference:

.. code-block:: python

    import torch
    from PIL import Image
    from yolo import create_model, bbox_nms

    # Load model
    model = create_model("v9-c", num_classes=80)
    model.load_state_dict(torch.load("weights/v9-c.pt"))
    model.eval()

    # Load and preprocess image
    image = Image.open("image.jpg")
    # ... preprocessing ...

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Apply NMS
    detections = bbox_nms(predictions, conf_thresh=0.5, iou_thresh=0.5)


NMS Configuration
-----------------

Non-Maximum Suppression parameters:

- ``min_confidence``: Minimum confidence threshold (default: 0.5)
- ``min_iou``: IoU threshold for NMS (default: 0.5)
