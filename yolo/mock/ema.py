"""
Mock EMA implementations for debugging.
"""

import copy
import math
from typing import Any, Dict, Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback

from .config import log_mock, should_log


class MockModelEMA:
    """
    Mock ModelEMA with detailed logging.

    Logs:
    - Effective decay at each update
    - Weight differences between model and EMA
    - BatchNorm statistics handling
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = updates

        for param in self.ema.parameters():
            param.requires_grad_(False)

        self._update_count = 0

        log_mock("ModelEMA", f"Initialized: decay={decay}, tau={tau}", force=True)

    def update(self, model: torch.nn.Module) -> None:
        """Update EMA weights with detailed logging."""
        self.updates += 1
        self._update_count += 1

        # Calculate effective decay
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        model_state = model.state_dict()

        # Track statistics
        num_ema_updated = 0
        num_bn_copied = 0
        max_weight_diff = 0.0

        with torch.no_grad():
            for name, ema_param in self.ema.state_dict().items():
                if name in model_state:
                    model_param = model_state[name]

                    # BatchNorm running statistics - COPY directly
                    if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                        ema_param.copy_(model_param)
                        num_bn_copied += 1
                    # Learnable parameters - EMA update
                    elif ema_param.dtype.is_floating_point:
                        # Track weight difference before update
                        diff = (ema_param - model_param).abs().max().item()
                        max_weight_diff = max(max_weight_diff, diff)

                        ema_param.mul_(d).add_(model_param, alpha=1 - d)
                        num_ema_updated += 1

        if should_log("ModelEMA", self._update_count):
            log_mock("ModelEMA", f"=== Update #{self._update_count} (total={self.updates}) ===")
            log_mock("ModelEMA", f"  Effective decay: {d:.6f}")
            log_mock("ModelEMA", f"  EMA params updated: {num_ema_updated}")
            log_mock("ModelEMA", f"  BN stats copied: {num_bn_copied}")
            log_mock("ModelEMA", f"  Max weight diff (before update): {max_weight_diff:.6f}")

            # Warn if decay is very low (weights changing too fast)
            if d < 0.5:
                log_mock("ModelEMA", f"  NOTE: Low effective decay ({d:.4f}) - EMA weights updating quickly")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_state_dict": self.ema.state_dict(),
            "decay": self.decay,
            "tau": self.tau,
            "updates": self.updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema.load_state_dict(state_dict["ema_state_dict"])
        self.decay = state_dict["decay"]
        self.tau = state_dict["tau"]
        self.updates = state_dict["updates"]


class MockEMACallback(Callback):
    """
    Mock EMA Callback with detailed logging.

    Logs:
    - When weights are swapped for validation
    - Differences between training and EMA weights
    """

    def __init__(
        self,
        decay: float = 0.9999,
        tau: float = 2000.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.tau = tau
        self.enabled = enabled

        self._ema: Optional[MockModelEMA] = None
        self._original_state_dict: Optional[Dict[str, Any]] = None
        self._swap_count = 0

        log_mock("EMACallback", f"Initialized: decay={decay}, tau={tau}, enabled={enabled}", force=True)

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not self.enabled:
            log_mock("EMACallback", "Disabled, skipping initialization", force=True)
            return

        if self._ema is not None:
            device = pl_module.device
            self._ema.ema.to(device)
            log_mock("EMACallback", f"Restored from checkpoint (updates={self._ema.updates})")
            return

        self._ema = MockModelEMA(
            model=pl_module,
            decay=self.decay,
            tau=self.tau,
            updates=0,
        )
        log_mock("EMACallback", "EMA model created")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.enabled or self._ema is None:
            return

        self._ema.update(pl_module)

    def on_validation_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not self.enabled or self._ema is None:
            return

        self._swap_count += 1

        # Save original weights
        self._original_state_dict = copy.deepcopy(pl_module.state_dict())

        # Load EMA weights
        pl_module.load_state_dict(self._ema.ema.state_dict())

        if should_log("EMACallback", self._swap_count):
            log_mock("EMACallback", f"=== Swap #{self._swap_count} to EMA weights for validation ===")

            # Compare a few key weights
            orig_state = self._original_state_dict
            ema_state = self._ema.ema.state_dict()

            sample_keys = list(orig_state.keys())[:5]
            for key in sample_keys:
                if orig_state[key].dtype.is_floating_point:
                    diff = (orig_state[key] - ema_state[key]).abs().mean().item()
                    log_mock("EMACallback", f"  {key}: mean diff = {diff:.6f}")

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not self.enabled or self._original_state_dict is None:
            return

        # Restore original weights
        pl_module.load_state_dict(self._original_state_dict)
        self._original_state_dict = None

        if should_log("EMACallback", self._swap_count):
            log_mock("EMACallback", "Restored training weights after validation")

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if not self.enabled or self._ema is None:
            return

        checkpoint["ema_callback"] = {
            "ema_state": self._ema.state_dict(),
            "decay": self.decay,
            "tau": self.tau,
        }
        log_mock("EMACallback", "Saved EMA state to checkpoint")

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        if "ema_callback" not in checkpoint:
            return

        ema_data = checkpoint["ema_callback"]

        if self._ema is None:
            saved_updates = 0
            if "ema_state" in ema_data:
                saved_updates = ema_data["ema_state"].get("updates", 0)

            self._ema = MockModelEMA(
                model=pl_module,
                decay=ema_data.get("decay", self.decay),
                tau=ema_data.get("tau", self.tau),
                updates=saved_updates,
            )

        if "ema_state" in ema_data:
            self._ema.load_state_dict(ema_data["ema_state"])
            log_mock("EMACallback", f"Loaded EMA from checkpoint (updates={self._ema.updates})")


class DisabledEMACallback(Callback):
    """
    Completely disabled EMA callback - does nothing.

    Use this to completely remove EMA from training for debugging.
    """

    def __init__(self, **kwargs):
        super().__init__()
        log_mock("DisabledEMA", "EMA completely disabled", force=True)
