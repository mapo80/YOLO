Quick Start
===========

.. note::
   This project uses **PyTorch Lightning** for training. All customizations are done via YAML config files or CLI overrides.

.. _QuickInstallYOLO:

Install YOLO
------------

Clone the repository and install the dependencies:

.. code-block:: bash

   git clone https://github.com/WongKinYiu/YOLO.git
   cd YOLO
   pip install -r requirements.txt

Train Model
-----------

To train the model using PyTorch Lightning:

.. code-block:: bash

   # Train with default config
   python -m yolo.cli fit --config yolo/config/experiment/default.yaml

   # Train with custom parameters
   python -m yolo.cli fit --config yolo/config/experiment/default.yaml \
       --model.model_config=v9-c \
       --data.batch_size=16 \
       --trainer.max_epochs=100

   # Quick debug run
   python -m yolo.cli fit --config yolo/config/experiment/debug.yaml

More details can be found in :doc:`../HOWTO`.

Inference
---------

To run inference on images:

.. code-block:: bash

   python examples/sample_inference.py --image path/to/image.jpg

   # With custom weights
   python examples/sample_inference.py --image path/to/image.jpg --weights weights/v9-c.pt

Validation
----------

To validate model performance:

.. code-block:: bash

   python -m yolo.cli validate --config yolo/config/experiment/default.yaml \
       --ckpt_path=runs/best.ckpt
