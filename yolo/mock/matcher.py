"""
Mock BoxMatcher for debugging target assignment.

This is the most critical component to debug - the soft target creation.
"""

from typing import Tuple

import torch
from torch import Tensor, tensor

from .config import log_mock, should_log


class MockBoxMatcher:
    """
    Mock BoxMatcher with detailed logging of the matching process.

    Logs:
    - Grid mask statistics
    - IoU matrix statistics
    - Classification matrix statistics
    - Target matrix (combined score)
    - TopK filtering results
    - normalize_term calculation (CRITICAL!)
    - Final soft targets statistics
    """

    def __init__(self, cfg, class_num: int, vec2box, reg_max: int) -> None:
        self.class_num = class_num
        self.vec2box = vec2box
        self.reg_max = reg_max
        for attr_name in cfg:
            setattr(self, attr_name, cfg[attr_name])

        self._call_count = 0

    def get_valid_matrix(self, target_bbox: Tensor):
        """Get boolean mask for anchors inside target bboxes."""
        x_min, y_min, x_max, y_max = target_bbox[:, :, None].unbind(3)
        anchors = self.vec2box.anchor_grid[None, None]
        anchors_x, anchors_y = anchors.unbind(dim=3)
        inside_x = (anchors_x >= x_min) & (anchors_x <= x_max)
        inside_y = (anchors_y >= y_min) & (anchors_y <= y_max)
        return inside_x & inside_y

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """Get predicted class probabilities for target classes."""
        predict_cls = predict_cls.transpose(1, 2)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox, target_bbox) -> Tensor:
        """Get IoU between targets and predictions."""
        from yolo.utils.bounding_box_utils import calculate_iou
        return calculate_iou(target_bbox, predict_bbox, self.iou).clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, grid_mask: Tensor, topk: int = 10) -> Tuple[Tensor, Tensor]:
        """Filter top-k targets for each anchor."""
        masked_target_matrix = grid_mask * target_matrix
        values, indices = masked_target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_mask = topk_targets > 0
        return topk_targets, topk_mask

    def ensure_one_anchor(self, target_matrix: Tensor, topk_mask: tensor) -> Tensor:
        """Ensure each target has at least one anchor."""
        values, indices = target_matrix.max(dim=-1)
        best_anchor_mask = torch.zeros_like(target_matrix, dtype=torch.bool)
        best_anchor_mask.scatter_(-1, index=indices[..., None], src=~best_anchor_mask)
        matched_anchor_num = torch.sum(topk_mask, dim=-1)
        target_without_anchor = (matched_anchor_num == 0) & (values > 0)
        topk_mask = torch.where(target_without_anchor[..., None], best_anchor_mask, topk_mask)
        return topk_mask

    def filter_duplicates(self, iou_mat: Tensor, topk_mask: Tensor):
        """Filter duplicates - one anchor maps to one target."""
        duplicates = (topk_mask.sum(1, keepdim=True) > 1).repeat([1, topk_mask.size(1), 1])
        masked_iou_mat = topk_mask * iou_mat
        best_indices = masked_iou_mat.argmax(1)[:, None, :]
        best_target_mask = torch.zeros_like(duplicates, dtype=torch.bool)
        best_target_mask.scatter_(1, index=best_indices, src=~best_target_mask)
        topk_mask = torch.where(duplicates, best_target_mask, topk_mask)
        unique_indices = topk_mask.to(torch.uint8).argmax(dim=1)
        return unique_indices[..., None], topk_mask.any(dim=1), topk_mask

    def __call__(self, target: Tensor, predict: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Match targets to anchors with detailed logging.

        This is the CRITICAL function where soft targets are created.
        The normalize_term calculation is the most suspicious part.
        """
        self._call_count += 1
        predict_cls, predict_bbox = predict

        # Handle empty targets
        n_targets = target.shape[1]
        if n_targets == 0:
            device = predict_bbox.device
            align_cls = torch.zeros_like(predict_cls, device=device)
            align_bbox = torch.zeros_like(predict_bbox, device=device)
            valid_mask = torch.zeros(predict_cls.shape[:2], dtype=bool, device=device)
            anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
            return anchor_matched_targets, valid_mask

        target_cls, target_bbox = target.split([1, 4], dim=-1)
        target_cls = target_cls.long().clamp(0)

        # Step 1: Get valid matrix
        grid_mask = self.get_valid_matrix(target_bbox)

        # Step 2: Get IoU matrix
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)

        # Step 3: Get classification matrix
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        # Step 4: Combined score
        target_matrix = (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"])

        # Step 5: TopK filtering
        topk_targets, topk_mask = self.filter_topk(target_matrix, grid_mask, topk=self.topk)

        # Step 6: Ensure each target has anchor
        topk_mask = self.ensure_one_anchor(target_matrix, topk_mask)

        # Step 7: Remove duplicates
        unique_indices, valid_mask, topk_mask = self.filter_duplicates(iou_mat, topk_mask)

        # Step 8: Build aligned targets
        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls_indices = torch.gather(target_cls, 1, unique_indices)
        align_cls = torch.zeros_like(align_cls_indices, dtype=torch.bool).repeat(1, 1, self.class_num)
        align_cls.scatter_(-1, index=align_cls_indices, src=~align_cls)

        # Step 9: NORMALIZE (SOFT TARGETS) - THIS IS CRITICAL!
        iou_mat_masked = iou_mat * topk_mask
        target_matrix_masked = target_matrix * topk_mask
        max_target = target_matrix_masked.amax(dim=-1, keepdim=True)
        max_iou = iou_mat_masked.amax(dim=-1, keepdim=True)

        # Original formula
        normalize_term = (target_matrix_masked / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)

        # Apply normalization to get soft labels
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        # Log statistics
        if should_log("BoxMatcher", self._call_count):
            log_mock("BoxMatcher", f"=== Call #{self._call_count} ===")
            log_mock("BoxMatcher", f"  Batch size: {target.shape[0]}")
            log_mock("BoxMatcher", f"  Num targets: {n_targets}")
            log_mock("BoxMatcher", f"  Num anchors: {predict_cls.shape[1]}")

            # Grid mask stats
            log_mock("BoxMatcher", f"  Grid mask: valid_anchors_per_target_mean={grid_mask.sum(-1).float().mean():.2f}")

            # IoU matrix stats
            iou_nonzero = iou_mat[iou_mat > 0]
            log_mock("BoxMatcher", f"  IoU matrix: min={iou_mat.min():.4f}, max={iou_mat.max():.4f}, mean={iou_nonzero.mean():.4f if len(iou_nonzero) > 0 else 0:.4f}")

            # Classification matrix stats
            cls_nonzero = cls_mat[cls_mat > 0]
            log_mock("BoxMatcher", f"  Cls matrix: min={cls_mat.min():.4f}, max={cls_mat.max():.4f}, mean={cls_nonzero.mean():.4f if len(cls_nonzero) > 0 else 0:.4f}")

            # Target matrix stats (combined score)
            tm_nonzero = target_matrix[target_matrix > 0]
            log_mock("BoxMatcher", f"  Target matrix: min={target_matrix.min():.4f}, max={target_matrix.max():.4f}, mean={tm_nonzero.mean():.4f if len(tm_nonzero) > 0 else 0:.4f}")

            # TopK stats
            log_mock("BoxMatcher", f"  TopK mask: valid_anchors={topk_mask.sum()}")

            # CRITICAL: normalize_term stats
            nt_nonzero = normalize_term[normalize_term > 0]
            log_mock("BoxMatcher", f"  [CRITICAL] normalize_term: min={normalize_term.min():.4f}, max={normalize_term.max():.4f}, mean={normalize_term.mean():.4f}")
            log_mock("BoxMatcher", f"  [CRITICAL] normalize_term nonzero: count={len(nt_nonzero)}, mean={nt_nonzero.mean():.4f if len(nt_nonzero) > 0 else 0:.4f}")

            # max_target and max_iou
            log_mock("BoxMatcher", f"  max_target: min={max_target.min():.4f}, max={max_target.max():.4f}")
            log_mock("BoxMatcher", f"  max_iou: min={max_iou.min():.4f}, max={max_iou.max():.4f}")

            # Final soft targets
            ac_nonzero = align_cls[align_cls > 0]
            log_mock("BoxMatcher", f"  [CRITICAL] align_cls (soft targets): min={align_cls.min():.4f}, max={align_cls.max():.4f}")
            log_mock("BoxMatcher", f"  [CRITICAL] align_cls nonzero: count={len(ac_nonzero)}, mean={ac_nonzero.mean():.4f if len(ac_nonzero) > 0 else 0:.4f}")

            # Valid mask
            log_mock("BoxMatcher", f"  Valid mask: {valid_mask.sum()} anchors matched")

            # Warnings
            if normalize_term.max() > 1.0:
                log_mock("BoxMatcher", f"  WARNING: normalize_term > 1.0 ({normalize_term.max():.4f})", force=True)
            if align_cls.max() > 1.0:
                log_mock("BoxMatcher", f"  WARNING: align_cls (soft targets) > 1.0 ({align_cls.max():.4f})", force=True)
            if valid_mask.sum() == 0 and n_targets > 0:
                log_mock("BoxMatcher", f"  WARNING: No anchors matched but targets exist!", force=True)

        anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
        return anchor_matched_targets, valid_mask
