import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.model.module import SPPELAN, ADown, CBLinear, Conv, Pool

STRIDE = 2
KERNEL_SIZE = 3
IN_CHANNELS = 64
OUT_CHANNELS = 128
NECK_CHANNELS = 64


def test_conv():
    conv = Conv(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = conv(x)
    assert out.shape == (1, OUT_CHANNELS, 64, 64)


def test_pool_max():
    pool = Pool("max", 2, stride=2)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = pool(x)
    assert out.shape == (1, IN_CHANNELS, 32, 32)


def test_pool_avg():
    pool = Pool("avg", 2, stride=2)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = pool(x)
    assert out.shape == (1, IN_CHANNELS, 32, 32)


def test_adown():
    adown = ADown(IN_CHANNELS, OUT_CHANNELS)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = adown(x)
    assert out.shape == (1, OUT_CHANNELS, 32, 32)


def test_cblinear():
    cblinear = CBLinear(IN_CHANNELS, [5, 5])
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    outs = cblinear(x)
    assert len(outs) == 2
    assert outs[0].shape == (1, 5, 64, 64)
    assert outs[1].shape == (1, 5, 64, 64)


def test_sppelan():
    sppelan = SPPELAN(IN_CHANNELS, OUT_CHANNELS, NECK_CHANNELS)
    x = torch.randn(1, IN_CHANNELS, 64, 64)
    out = sppelan(x)
    assert out.shape == (1, OUT_CHANNELS, 64, 64)


# =============================================================================
# C1: Detection Head use_group=False Tests
# =============================================================================


def test_detection_default_use_group_is_false():
    """
    Test C1 fix: Detection head should default to use_group=False.

    yolov9-official Detect class does NOT use grouped convolutions.
    This ensures full capacity in the detection head.
    """
    from yolo.model.module import Detection
    import inspect

    # Check the default value in the signature
    sig = inspect.signature(Detection.__init__)
    use_group_param = sig.parameters.get("use_group")

    assert use_group_param is not None, "Detection should have use_group parameter"
    assert use_group_param.default is False, (
        f"Detection.use_group should default to False (aligned with yolov9-official), "
        f"but got {use_group_param.default}"
    )


def test_detection_no_groups_when_use_group_false():
    """
    Test that Detection head uses groups=1 when use_group=False.
    """
    from yolo.model.module import Detection

    # Detection expects in_channels as tuple (first_neck, in_channel)
    # first_neck is used to determine intermediate channel sizes
    # in_channel is the actual input channels for this head
    first_neck = 64
    in_channel = 128
    in_channels = (first_neck, in_channel)
    num_classes = 80
    reg_max = 16

    detect = Detection(in_channels, num_classes, reg_max=reg_max, use_group=False)

    # Check that conv layers don't use groups > 1
    for name, module in detect.named_modules():
        if hasattr(module, 'groups'):
            assert module.groups == 1, (
                f"Module {name} has groups={module.groups}, expected 1 with use_group=False"
            )


def test_detection_uses_groups_when_use_group_true():
    """
    Test that Detection head uses grouped convs when use_group=True.
    """
    from yolo.model.module import Detection

    first_neck = 64
    in_channel = 128
    in_channels = (first_neck, in_channel)
    num_classes = 80
    reg_max = 16

    detect = Detection(in_channels, num_classes, reg_max=reg_max, use_group=True)

    # With use_group=True, some convs should have groups > 1
    has_grouped_conv = False
    for name, module in detect.named_modules():
        if hasattr(module, 'groups') and module.groups > 1:
            has_grouped_conv = True
            break

    assert has_grouped_conv, (
        "Detection with use_group=True should have at least one grouped conv"
    )


def test_detection_forward_shape():
    """
    Test Detection forward pass output shape.
    """
    from yolo.model.module import Detection

    # Detection is for a single scale, not multi-scale
    first_neck = 64
    in_channel = 128
    in_channels = (first_neck, in_channel)
    num_classes = 80
    reg_max = 16
    batch_size = 2

    detect = Detection(in_channels, num_classes, reg_max=reg_max, use_group=False)

    # Single scale input
    x = torch.randn(batch_size, in_channel, 40, 40)

    cls, anc, box = detect(x)

    # cls: [B, num_classes, H, W]
    # anc: [B, reg_max, 4, H, W] - note: reg_max comes before 4 (ltrb)
    # box: [B, 4, H, W]
    assert cls.shape == (batch_size, num_classes, 40, 40), f"cls shape mismatch: {cls.shape}"
    assert anc.shape == (batch_size, reg_max, 4, 40, 40), f"anc shape mismatch: {anc.shape}"
    assert box.shape == (batch_size, 4, 40, 40), f"box shape mismatch: {box.shape}"
