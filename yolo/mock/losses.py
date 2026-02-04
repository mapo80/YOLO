"""
Mock loss functions for debugging YOLO training.

Each mock logs detailed statistics about inputs/outputs to help diagnose issues.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import log_mock, should_log


class MockBCELoss(nn.Module):
    """
    Mock BCE Loss with detailed logging.

    Logs:
    - Input shapes
    - Target statistics (min, max, mean, sum, non-zero count)
    - Prediction statistics (logits and sigmoid)
    - Loss value before and after normalization
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self._call_count = 0

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Tensor:
        self._call_count += 1

        # Compute raw loss
        loss_raw = self.bce(predicts_cls, targets_cls)
        loss_normalized = loss_raw.sum() / cls_norm

        # Log statistics
        if should_log("BCELoss", self._call_count):
            pred_sigmoid = predicts_cls.sigmoid()
            targets_nonzero = targets_cls[targets_cls > 0]

            log_mock("BCELoss", f"=== Call #{self._call_count} ===")
            log_mock("BCELoss", f"  Shape: predicts={predicts_cls.shape}, targets={targets_cls.shape}")
            log_mock("BCELoss", f"  Predicts logits: min={predicts_cls.min():.4f}, max={predicts_cls.max():.4f}, mean={predicts_cls.mean():.4f}")
            log_mock("BCELoss", f"  Predicts sigmoid: min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}")
            log_mock("BCELoss", f"  Targets: min={targets_cls.min():.4f}, max={targets_cls.max():.4f}, mean={targets_cls.mean():.4f}")
            log_mock("BCELoss", f"  Targets sum: {targets_cls.sum():.4f}")
            nz_mean = targets_nonzero.mean().item() if len(targets_nonzero) > 0 else 0.0
            log_mock("BCELoss", f"  Targets non-zero: count={len(targets_nonzero)}, mean={nz_mean:.4f}")
            log_mock("BCELoss", f"  cls_norm: {cls_norm:.4f}")
            log_mock("BCELoss", f"  Loss raw sum: {loss_raw.sum():.4f}")
            log_mock("BCELoss", f"  Loss normalized: {loss_normalized:.4f}")

            # Alert on suspicious values
            if targets_cls.max() > 1.0:
                log_mock("BCELoss", f"  WARNING: targets_cls.max() > 1.0 ({targets_cls.max():.4f})", force=True)
            if cls_norm < 1.0:
                log_mock("BCELoss", f"  WARNING: cls_norm < 1.0 ({cls_norm:.4f})", force=True)
            if predicts_cls.max() < -5.0:
                log_mock("BCELoss", f"  WARNING: predicts very negative (max={predicts_cls.max():.4f})", force=True)

        return loss_normalized


class MockBoxLoss(nn.Module):
    """
    Mock Box Loss with detailed logging.

    Logs:
    - Number of valid boxes
    - IoU statistics
    - box_norm statistics
    - Loss value
    """

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def forward(
        self,
        predicts_bbox: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        from yolo.utils.bounding_box_utils import calculate_iou

        self._call_count += 1

        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        if len(picked_predict) == 0:
            return torch.tensor(0.0, device=predicts_bbox.device)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_weighted = (loss_iou * box_norm).sum() / cls_norm

        # Log statistics
        if should_log("BoxLoss", self._call_count):
            log_mock("BoxLoss", f"=== Call #{self._call_count} ===")
            log_mock("BoxLoss", f"  Valid boxes: {valid_masks.sum()}")
            log_mock("BoxLoss", f"  IoU: min={iou.min():.4f}, max={iou.max():.4f}, mean={iou.mean():.4f}")
            log_mock("BoxLoss", f"  box_norm: min={box_norm.min():.4f}, max={box_norm.max():.4f}, mean={box_norm.mean():.4f}")
            log_mock("BoxLoss", f"  cls_norm: {cls_norm:.4f}")
            log_mock("BoxLoss", f"  Loss (1-IoU) mean: {loss_iou.mean():.4f}")
            log_mock("BoxLoss", f"  Loss weighted: {loss_weighted:.4f}")

        return loss_weighted


class MockDFLoss(nn.Module):
    """
    Mock DFL Loss with detailed logging.

    Logs:
    - Target distance statistics
    - Prediction statistics
    - Loss value
    """

    def __init__(self, vec2box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max
        self._call_count = 0

    def forward(
        self,
        predicts_anc: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        self._call_count += 1

        # valid_masks: [B, num_anchors]
        # predicts_anc: [B, num_anchors, 4, reg_max]
        # targets_bbox: [B, num_anchors, 4]

        # Expand valid_masks for both (shape [B, num_anchors, 4])
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)

        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(
            ((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1
        ).clamp(0, self.reg_max - 1.01)
        picked_targets = targets_dist[valid_bbox].view(-1)
        # predicts_anc indexing: select valid [B, num_anchors, 4] -> then .view(-1, reg_max)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        if len(picked_predict) == 0:
            return torch.tensor(0.0, device=predicts_anc.device)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_weighted = (loss_dfl * box_norm).sum() / cls_norm

        # Log statistics
        if should_log("DFLoss", self._call_count):
            log_mock("DFLoss", f"=== Call #{self._call_count} ===")
            log_mock("DFLoss", f"  reg_max: {self.reg_max}")
            log_mock("DFLoss", f"  Target dist: min={picked_targets.min():.4f}, max={picked_targets.max():.4f}, mean={picked_targets.mean():.4f}")
            log_mock("DFLoss", f"  Valid count: {len(picked_predict)}")
            log_mock("DFLoss", f"  DFL loss mean: {loss_dfl.mean():.4f}")
            log_mock("DFLoss", f"  Loss weighted: {loss_weighted:.4f}")

            # Check for out-of-range values
            if picked_targets.max() >= self.reg_max - 1:
                log_mock("DFLoss", f"  WARNING: targets near reg_max limit ({picked_targets.max():.4f})", force=True)

        return loss_weighted


class MockYOLOLoss(nn.Module):
    """
    Mock YOLO Loss that wraps all component losses with detailed logging.

    Logs:
    - All input shapes
    - Intermediate values (targets_cls, cls_norm, box_norm)
    - Individual loss components
    - Total loss
    """

    def __init__(
        self,
        vec2box,
        class_num: int = 80,
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        cls_pos_weight: float = 1.0,
        cls_loss_type: str = "bce",
        cls_vfl_alpha: float = 0.75,
        cls_vfl_gamma: float = 2.0,
        matcher_topk: int = 13,  # Aligned with yolov9-official (was 10)
        matcher_iou_weight: float = 6.0,
        matcher_cls_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.class_num = class_num
        self.vec2box = vec2box
        self.reg_max = reg_max

        # Loss weights
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

        # Use mock losses
        self.cls_loss = MockBCELoss()
        self.dfl_loss = MockDFLoss(vec2box, reg_max)
        self.box_loss = MockBoxLoss()

        # Import original matcher (or use mock if configured)
        from yolo.utils.bounding_box_utils import BoxMatcher
        from dataclasses import dataclass

        @dataclass
        class MatcherConfig:
            iou: str = "ciou"
            topk: int = 10
            factor: dict = None

            def __iter__(self):
                return iter(["iou", "topk", "factor"])

            def __getitem__(self, key):
                return getattr(self, key)

        cfg = MatcherConfig(
            iou="ciou",
            topk=matcher_topk,
            factor={"iou": matcher_iou_weight, "cls": matcher_cls_weight},
        )
        self.matcher = BoxMatcher(cfg, class_num, vec2box, reg_max)

        self._call_count = 0

    def _separate_anchor(self, anchors: Tensor) -> Tuple[Tensor, Tensor]:
        """Separate anchor predictions into class and bbox components."""
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: List,
    ) -> Tuple[Tensor, Dict[str, float]]:
        self._call_count += 1

        # Convert targets to tensor format
        targets_tensor = self._prepare_targets(targets)

        # Get main predictions
        main_preds = outputs["Main"]
        predicts_cls, predicts_anc, predicts_box = self.vec2box(main_preds)

        # Match targets to anchors (returns SOFT targets)
        align_targets, valid_masks = self.matcher(
            targets_tensor, (predicts_cls.detach(), predicts_box.detach())
        )

        targets_cls, targets_bbox = self._separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        # cls_norm = sum of soft target values (original YOLOv9 design)
        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        # Log before computing losses
        if should_log("YOLOLoss", self._call_count):
            log_mock("YOLOLoss", f"=== Call #{self._call_count} ===")
            log_mock("YOLOLoss", f"  Batch size: {predicts_cls.shape[0]}")
            log_mock("YOLOLoss", f"  Num anchors: {predicts_cls.shape[1]}")
            log_mock("YOLOLoss", f"  Num targets: {targets_tensor.shape[1]}")
            log_mock("YOLOLoss", f"  Valid anchors: {valid_masks.sum()}")
            log_mock("YOLOLoss", f"  [CRITICAL] targets_cls sum: {targets_cls.sum():.4f}")
            log_mock("YOLOLoss", f"  [CRITICAL] targets_cls non-zero count: {(targets_cls > 0).sum()}")
            nz = targets_cls[targets_cls > 0]
            if len(nz) > 0:
                log_mock("YOLOLoss", f"  [CRITICAL] targets_cls non-zero: min={nz.min():.4f}, max={nz.max():.4f}, mean={nz.mean():.4f}")
            log_mock("YOLOLoss", f"  cls_norm: {cls_norm:.4f}")
            log_mock("YOLOLoss", f"  box_norm sum: {box_norm.sum():.4f}")
            # Log predicts_cls stats
            log_mock("YOLOLoss", f"  [CRITICAL] predicts_cls (logits): min={predicts_cls.min():.4f}, max={predicts_cls.max():.4f}, mean={predicts_cls.mean():.4f}")

        # Compute losses
        loss_cls = self.cls_loss(predicts_cls, targets_cls, cls_norm)
        loss_box = self.box_loss(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        loss_dfl = self.dfl_loss(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        # Apply weights
        loss_cls_weighted = self.cls_weight * loss_cls
        loss_box_weighted = self.box_weight * loss_box
        loss_dfl_weighted = self.dfl_weight * loss_dfl

        total_loss = loss_cls_weighted + loss_box_weighted + loss_dfl_weighted

        # Handle auxiliary head if present
        if "AUX" in outputs:
            aux_preds = outputs["AUX"]
            aux_cls, aux_anc, aux_box = self.vec2box(aux_preds)

            aux_align_targets, aux_valid_masks = self.matcher(
                targets_tensor, (aux_cls.detach(), aux_box.detach())
            )

            aux_targets_cls, aux_targets_bbox = self._separate_anchor(aux_align_targets)
            aux_box = aux_box / self.vec2box.scaler[None, :, None]

            aux_cls_norm = max(aux_targets_cls.sum(), 1)
            aux_box_norm = aux_targets_cls.sum(-1)[aux_valid_masks]

            aux_loss_cls = self.cls_loss(aux_cls, aux_targets_cls, aux_cls_norm)
            aux_loss_box = self.box_loss(aux_box, aux_targets_bbox, aux_valid_masks, aux_box_norm, aux_cls_norm)
            aux_loss_dfl = self.dfl_loss(aux_anc, aux_targets_bbox, aux_valid_masks, aux_box_norm, aux_cls_norm)

            # Auxiliary loss with 0.25 weight
            aux_weight = 0.25
            total_loss = total_loss + aux_weight * (
                self.cls_weight * aux_loss_cls +
                self.box_weight * aux_loss_box +
                self.dfl_weight * aux_loss_dfl
            )

        if should_log("YOLOLoss", self._call_count):
            log_mock("YOLOLoss", f"  loss_cls (weighted): {loss_cls_weighted:.4f}")
            log_mock("YOLOLoss", f"  loss_box (weighted): {loss_box_weighted:.4f}")
            log_mock("YOLOLoss", f"  loss_dfl (weighted): {loss_dfl_weighted:.4f}")
            log_mock("YOLOLoss", f"  total_loss: {total_loss:.4f}")

        loss_dict = {
            "box_loss": loss_box_weighted.detach(),
            "cls_loss": loss_cls_weighted.detach(),
            "dfl_loss": loss_dfl_weighted.detach(),
        }

        return total_loss, loss_dict

    def _prepare_targets(self, targets: List) -> Tensor:
        """Convert targets to tensor format [batch, max_targets, 5]."""
        batch_size = len(targets)
        device = self.vec2box.anchor_grid.device

        max_targets = 0
        processed_targets = []

        for target in targets:
            if isinstance(target, Tensor):
                processed_targets.append(target)
                max_targets = max(max_targets, len(target))
            elif isinstance(target, dict):
                boxes = target.get("boxes", torch.zeros((0, 4)))
                labels = target.get("labels", torch.zeros((0,)))
                if len(boxes) > 0:
                    t = torch.cat([labels.unsqueeze(-1).float(), boxes.float()], dim=-1)
                else:
                    t = torch.zeros((0, 5))
                processed_targets.append(t)
                max_targets = max(max_targets, len(t))
            elif isinstance(target, list):
                boxes = []
                for ann in target:
                    if "bbox" in ann:
                        x, y, w, h = ann["bbox"]
                        boxes.append([ann.get("category_id", 0), x, y, x + w, y + h])
                if boxes:
                    t = torch.tensor(boxes, dtype=torch.float32)
                else:
                    t = torch.zeros((0, 5))
                processed_targets.append(t)
                max_targets = max(max_targets, len(t))
            else:
                processed_targets.append(torch.zeros((0, 5)))

        if max_targets == 0:
            return torch.zeros((batch_size, 0, 5), device=device)

        padded_targets = torch.zeros((batch_size, max_targets, 5), device=device)
        for i, t in enumerate(processed_targets):
            if len(t) > 0:
                padded_targets[i, :len(t)] = t.to(device)

        return padded_targets
