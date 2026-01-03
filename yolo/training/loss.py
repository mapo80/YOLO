"""
YOLO Loss Functions - Clean implementation for Lightning training.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou


class BCELoss(nn.Module):
    """Binary Cross Entropy loss for classification."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Tensor:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    """Box regression loss using CIoU."""

    def forward(
        self,
        predicts_bbox: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    """Distribution Focal Loss for box regression."""

    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self,
        predicts_anc: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(
            ((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1
        ).clamp(0, self.reg_max - 1.01)
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOLoss(nn.Module):
    """
    YOLO Loss function combining classification, box regression and DFL losses.

    Args:
        vec2box: Vector to box converter
        class_num: Number of classes
        reg_max: Maximum regression value for DFL
        box_weight: Weight for box loss
        cls_weight: Weight for classification loss
        dfl_weight: Weight for DFL loss
        matcher_topk: Top-k for anchor matching
        matcher_iou_weight: IoU weight in matching
        matcher_cls_weight: Classification weight in matching
    """

    def __init__(
        self,
        vec2box: Vec2Box,
        class_num: int = 80,
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        matcher_topk: int = 10,
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

        # Loss functions
        self.cls_loss = BCELoss()
        self.dfl_loss = DFLoss(vec2box, reg_max)
        self.box_loss = BoxLoss()

        # Matcher config
        self.matcher = self._create_matcher(
            vec2box, class_num, reg_max, matcher_topk, matcher_iou_weight, matcher_cls_weight
        )

    def _create_matcher(
        self,
        vec2box: Vec2Box,
        class_num: int,
        reg_max: int,
        topk: int,
        iou_weight: float,
        cls_weight: float,
    ) -> BoxMatcher:
        """Create box matcher with configuration."""
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
            topk=topk,
            factor={"iou": iou_weight, "cls": cls_weight},
        )
        return BoxMatcher(cfg, class_num, vec2box, reg_max)

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
        """
        Compute YOLO loss.

        Args:
            outputs: Model outputs dict with "Main" and optionally "AUX" keys
            targets: List of target annotations

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss values
        """
        # Convert targets to tensor format
        targets_tensor = self._prepare_targets(targets)

        # Get main predictions
        main_preds = outputs["Main"]
        predicts_cls, predicts_anc, predicts_box = self.vec2box(main_preds)

        # Match targets to anchors
        align_targets, valid_masks = self.matcher(
            targets_tensor, (predicts_cls.detach(), predicts_box.detach())
        )

        targets_cls, targets_bbox = self._separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

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

        loss_dict = {
            "box_loss": loss_box_weighted.detach(),
            "cls_loss": loss_cls_weighted.detach(),
            "dfl_loss": loss_dfl_weighted.detach(),
        }

        return total_loss, loss_dict

    def _prepare_targets(self, targets: List) -> Tensor:
        """
        Convert targets to tensor format [batch, max_targets, 5].

        Expected format: [class, x1, y1, x2, y2]
        """
        batch_size = len(targets)
        device = self.vec2box.anchor_grid.device

        # Find max targets in batch
        max_targets = 0
        processed_targets = []

        for target in targets:
            if isinstance(target, Tensor):
                processed_targets.append(target)
                max_targets = max(max_targets, len(target))
            elif isinstance(target, dict):
                # Dict format from transforms v2
                boxes = target.get("boxes", torch.zeros((0, 4)))
                labels = target.get("labels", torch.zeros((0,)))
                if len(boxes) > 0:
                    t = torch.cat([labels.unsqueeze(-1).float(), boxes.float()], dim=-1)
                else:
                    t = torch.zeros((0, 5))
                processed_targets.append(t)
                max_targets = max(max_targets, len(t))
            elif isinstance(target, list):
                # List of COCO annotations
                boxes = []
                labels = []
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

        # Pad to same size
        if max_targets == 0:
            return torch.zeros((batch_size, 0, 5), device=device)

        padded_targets = torch.zeros((batch_size, max_targets, 5), device=device)
        for i, t in enumerate(processed_targets):
            if len(t) > 0:
                padded_targets[i, :len(t)] = t.to(device)

        return padded_targets
