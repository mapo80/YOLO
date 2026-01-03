Install YOLO
============

This guide will help you set up YOLO on your machine.
We recommend starting with `GitHub Settings <#git-github>`_ for more flexible customization.
If you are planning to perform inference only or require a simple customization, you can choose to install via `PyPI <#pypi-pip-install>`_.

Torch Requirements
-------------------

The following table summarizes the torch requirements for different operating systems and hardware configurations:


.. tabs::

   .. tab:: Linux

      .. tabs::

         .. tab:: CUDA

            PyTorch: 1.12+

         .. tab:: CPU

            PyTorch: 1.12+

   .. tab:: MacOS

      .. tabs::

         .. tab:: MPS

            PyTorch: 2.2+
         .. tab:: CPU
            PyTorch: 2.2+
   .. tab:: Windows

      .. tabs::

         .. tab:: CUDA

            [WIP]

         .. tab:: CPU

            [WIP]


Git & GitHub
------------

First, Clone the repository:

.. code-block:: bash

   git clone https://github.com/WongKinYiu/YOLO.git

Alternatively, you can directly download the repository via this `link <https://github.com/WongKinYiu/YOLO/archive/refs/heads/main.zip>`_.

Next, install the required packages:

.. code-block:: bash

    pip install -r requirements.txt

PyPI (pip install)
------------------

.. note::
    Due to the :guilabel:`yolo` this name already being occupied in the PyPI library, we are still determining the package name.
    Currently, we provide an alternative way to install via the GitHub repository. Ensure your shell has `git` and `pip3` (or `pip`).

To install YOLO via GitHub:

.. code-block:: bash

   pip install git+https://github.com/WongKinYiu/YOLO.git

