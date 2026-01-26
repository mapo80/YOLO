"""
Mock Vec2Box for debugging prediction conversion.
"""

from typing import List

import torch
from einops import rearrange
from torch import Tensor

from .config import log_mock, should_log


class MockVec2Box:
    """
    Mock Vec2Box with detailed logging of prediction conversion.

    Logs:
    - Input shapes from each detection head
    - Anchor grid statistics
    - Output shapes and statistics
    """

    def __init__(self, model, anchor_cfg, image_size, device):
        from yolo.utils.bounding_box_utils import generate_anchors

        self.device = device

        if hasattr(anchor_cfg, "strides"):
            self.strides = anchor_cfg.strides
        else:
            self.strides = self._create_auto_anchor(model, image_size)

        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

        self._call_count = 0

        log_mock("Vec2Box", f"Initialized: image_size={image_size}, strides={self.strides}", force=True)
        log_mock("Vec2Box", f"  anchor_grid shape: {self.anchor_grid.shape}", force=True)
        log_mock("Vec2Box", f"  scaler shape: {self.scaler.shape}", force=True)

    def _create_auto_anchor(self, model, image_size):
        W, H = image_size
        dummy_input = torch.zeros(1, 3, H, W)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(W // anchor_num[1])
        return strides

    def update(self, image_size):
        """Update anchor grid for new image size."""
        if self.image_size == image_size:
            return
        from yolo.utils.bounding_box_utils import generate_anchors
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(self.device), scaler.to(self.device)
        log_mock("Vec2Box", f"Updated: new image_size={image_size}")

    def __call__(self, predicts: List) -> tuple:
        """
        Convert predictions from detection heads to usable format.

        Input: list of (pred_cls, pred_anc, pred_box) tuples from each stride
        Output: (preds_cls, preds_anc, preds_box) concatenated across strides
        """
        self._call_count += 1

        preds_cls, preds_anc, preds_box = [], [], []

        for layer_idx, layer_output in enumerate(predicts):
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))

            if should_log("Vec2Box", self._call_count) and layer_idx == 0:
                log_mock("Vec2Box", f"=== Call #{self._call_count} ===")
                log_mock("Vec2Box", f"  Num layers: {len(predicts)}")

        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        # Convert LTRB to xyxy
        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)

        if should_log("Vec2Box", self._call_count):
            log_mock("Vec2Box", f"  Output shapes:")
            log_mock("Vec2Box", f"    preds_cls: {preds_cls.shape}")
            log_mock("Vec2Box", f"    preds_anc: {preds_anc.shape}")
            log_mock("Vec2Box", f"    preds_box: {preds_box.shape}")

            # Class predictions stats
            log_mock("Vec2Box", f"  preds_cls (logits): min={preds_cls.min():.4f}, max={preds_cls.max():.4f}, mean={preds_cls.mean():.4f}")

            # Box predictions stats
            log_mock("Vec2Box", f"  preds_box: min={preds_box.min():.4f}, max={preds_box.max():.4f}")

            # Check for NaN/Inf
            if torch.isnan(preds_cls).any():
                log_mock("Vec2Box", f"  WARNING: NaN in preds_cls!", force=True)
            if torch.isnan(preds_box).any():
                log_mock("Vec2Box", f"  WARNING: NaN in preds_box!", force=True)
            if torch.isinf(preds_cls).any():
                log_mock("Vec2Box", f"  WARNING: Inf in preds_cls!", force=True)

        return preds_cls, preds_anc, preds_box
