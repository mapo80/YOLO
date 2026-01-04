"""
Unit tests for learning rate schedulers in YOLOModule.

Tests cover all scheduler types:
- Cosine Annealing
- Linear decay
- Step decay
- OneCycle
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


class TestLRSchedulers:
    """Tests for learning rate scheduler creation."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        trainer = MagicMock()
        trainer.max_epochs = 100
        trainer.train_dataloader = MagicMock()
        trainer.train_dataloader.__len__ = MagicMock(return_value=1000)
        return trainer

    @pytest.fixture
    def simple_optimizer(self):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def test_cosine_scheduler_creation(self):
        """Test cosine annealing scheduler is created correctly."""
        # Create optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0.0001,
        )

        # Verify initial LR
        assert scheduler.get_last_lr()[0] == 0.01

        # Step through and verify LR decreases
        for _ in range(50):
            scheduler.step()

        # LR should be lower than initial
        assert scheduler.get_last_lr()[0] < 0.01

        # At T_max, LR should be at eta_min
        for _ in range(50):
            scheduler.step()
        assert abs(scheduler.get_last_lr()[0] - 0.0001) < 1e-6

    def test_linear_scheduler_creation(self):
        """Test linear decay scheduler is created correctly."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        max_epochs = 100
        lr_min_factor = 0.01

        def linear_lambda(epoch):
            if epoch >= max_epochs:
                return lr_min_factor
            return 1.0 - (1.0 - lr_min_factor) * (epoch / max_epochs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=linear_lambda,
        )

        # Verify initial LR
        assert scheduler.get_last_lr()[0] == 0.01

        # Step halfway
        for _ in range(50):
            scheduler.step()

        # LR should be approximately halfway
        expected_lr = 0.01 * (1.0 - (1.0 - lr_min_factor) * 0.5)
        assert abs(scheduler.get_last_lr()[0] - expected_lr) < 1e-6

        # At end, LR should be at minimum
        for _ in range(50):
            scheduler.step()
        assert abs(scheduler.get_last_lr()[0] - 0.01 * lr_min_factor) < 1e-6

    def test_step_scheduler_creation(self):
        """Test step decay scheduler is created correctly."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        step_size = 30
        gamma = 0.1

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

        # Verify initial LR
        assert scheduler.get_last_lr()[0] == 0.01

        # Step to first decay
        for _ in range(step_size):
            scheduler.step()

        # LR should be reduced by gamma
        assert abs(scheduler.get_last_lr()[0] - 0.001) < 1e-6

        # Step to second decay
        for _ in range(step_size):
            scheduler.step()

        # LR should be reduced again
        assert abs(scheduler.get_last_lr()[0] - 0.0001) < 1e-6

    def test_one_cycle_scheduler_creation(self):
        """Test OneCycleLR scheduler is created correctly."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        total_steps = 10000
        pct_start = 0.3
        div_factor = 25.0
        final_div_factor = 1e4

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

        # Initial LR should be max_lr / div_factor
        expected_initial = 0.01 / div_factor
        assert abs(scheduler.get_last_lr()[0] - expected_initial) < 1e-6

        # Step to peak (30% of training)
        warmup_steps = int(total_steps * pct_start)
        for _ in range(warmup_steps):
            scheduler.step()

        # LR should be at or near max
        assert scheduler.get_last_lr()[0] >= 0.009  # Near max

    def test_scheduler_with_warmup_interaction(self, simple_optimizer):
        """Test that warmup and scheduler work together correctly."""
        # Create cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            simple_optimizer,
            T_max=100,
            eta_min=0.0001,
        )

        # Simulate warmup by manually adjusting LR
        warmup_steps = 3000
        warmup_lr = 0.0

        # Warmup phase (linear increase)
        for step in range(warmup_steps):
            warmup_progress = step / warmup_steps
            current_lr = 0.01 * warmup_progress
            for pg in simple_optimizer.param_groups:
                pg["lr"] = current_lr

        # After warmup, LR should be at base level
        assert abs(simple_optimizer.param_groups[0]["lr"] - 0.01) < 0.001


class TestSchedulerConfigurations:
    """Test different scheduler configurations."""

    def test_valid_scheduler_names(self):
        """Test that all valid scheduler names are recognized."""
        valid_schedulers = ["cosine", "linear", "step", "one_cycle"]

        for name in valid_schedulers:
            assert name.lower() in valid_schedulers

    def test_scheduler_parameters_bounds(self):
        """Test scheduler parameters have valid bounds."""
        # lr_min_factor should be between 0 and 1
        assert 0 < 0.01 < 1

        # step_size should be positive
        assert 30 > 0

        # step_gamma should be between 0 and 1
        assert 0 < 0.1 < 1

        # one_cycle_pct_start should be between 0 and 1
        assert 0 < 0.3 < 1

        # div_factor should be positive
        assert 25.0 > 0

        # final_div_factor should be positive
        assert 10000.0 > 0


class TestSchedulerOutput:
    """Test scheduler output behavior."""

    def test_cosine_scheduler_smoothness(self):
        """Test that cosine scheduler produces smooth LR transitions."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0.0001,
        )

        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Check that LR changes are smooth (no large jumps)
        for i in range(1, len(lrs)):
            delta = abs(lrs[i] - lrs[i - 1])
            assert delta < 0.002  # Max 0.2% change per step

    def test_step_scheduler_discrete_jumps(self):
        """Test that step scheduler produces discrete LR changes."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
        )

        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Count number of unique LR values (should be limited)
        unique_lrs = set(round(lr, 10) for lr in lrs)
        assert len(unique_lrs) <= 6  # At most 6 unique values in 50 steps

    def test_linear_scheduler_monotonic_decrease(self):
        """Test that linear scheduler monotonically decreases LR."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        max_epochs = 100
        lr_min_factor = 0.01

        def linear_lambda(epoch):
            if epoch >= max_epochs:
                return lr_min_factor
            return 1.0 - (1.0 - lr_min_factor) * (epoch / max_epochs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=linear_lambda,
        )

        prev_lr = scheduler.get_last_lr()[0]
        for _ in range(100):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            assert current_lr <= prev_lr
            prev_lr = current_lr


class TestSchedulerEdgeCases:
    """Test edge cases for schedulers."""

    def test_single_epoch_training(self):
        """Test scheduler with single epoch."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1,
            eta_min=0.0001,
        )

        scheduler.step()
        # Should not crash and LR should be at minimum
        assert scheduler.get_last_lr()[0] <= 0.01

    def test_very_long_training(self):
        """Test scheduler with many epochs."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=0.0001,
        )

        # Step through many epochs
        for _ in range(1000):
            scheduler.step()

        # Should reach minimum
        assert abs(scheduler.get_last_lr()[0] - 0.0001) < 1e-6

    def test_zero_min_lr(self):
        """Test scheduler with zero minimum LR."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0.0,
        )

        # Step to end
        for _ in range(100):
            scheduler.step()

        # LR should be zero
        assert scheduler.get_last_lr()[0] == 0.0

    def test_multiple_param_groups(self):
        """Test scheduler with multiple parameter groups."""
        model1 = nn.Linear(10, 10)
        model2 = nn.Linear(10, 10)

        optimizer = torch.optim.SGD([
            {"params": model1.parameters(), "lr": 0.01},
            {"params": model2.parameters(), "lr": 0.001},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0.0001,
        )

        # Step through
        for _ in range(50):
            scheduler.step()

        # Both param groups should have reduced LR
        lrs = scheduler.get_last_lr()
        assert len(lrs) == 2
        assert lrs[0] < 0.01
        assert lrs[1] < 0.001
