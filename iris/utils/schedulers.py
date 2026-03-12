"""Scheduler Classes"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler


@dataclass
class ChainedSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: ChainedScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    gamma: float = 0.33
    """The learning rate decay factor."""
    milestones: tuple = (0.5, 0.75, 0.9)
    """The milestone steps at which to decay the learning rate."""


class ChainedScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: ChainedSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma,
        )

        scheduler = lr_scheduler.ChainedScheduler(
            [
                lr_scheduler.LinearLR(optimizer=optimizer, 
                                      start_factor=0.01, 
                                      total_iters=100),
                lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                         milestones=[int(m * self.config.max_steps) for m in self.config.milestones], 
                                         gamma=self.config.gamma),
            ]
        )

        return scheduler

@dataclass
class OneCycleLRSchedulerConfig(SchedulerConfig):
    """Config for one cycle lr scheduler"""

    _target: Type = field(default_factory=lambda: OneCycleLRScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The minimum learning rate."""
    lr_max: float = 1e-3
    """The maximum learning rate."""

class OneCycleLRScheduler(Scheduler):
    """One cycle lr scheduler"""

    config: OneCycleLRSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.config.lr_max, total_steps=self.config.max_steps)
        return scheduler


@dataclass
class CosineAnnealingSchedulerConfig(SchedulerConfig):
    """Config for cosine annealing lr scheduler"""

    _target: Type = field(default_factory=lambda: CosineAnnealingScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    eta_min: float = 1e-5
    """The minimum learning rate."""

class CosineAnnealingScheduler(Scheduler):
    """Cosine annealing lr scheduler"""

    config: CosineAnnealingSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.config.max_steps, eta_min=self.config.eta_min)
        return scheduler


@dataclass
class ExponentialSchedulerConfig(SchedulerConfig):
    """Config for exponential lr scheduler"""

    _target: Type = field(default_factory=lambda: ExponentialScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    min_lr: float = 1e-5
    """The learning rate decay factor."""
    init_lr: float = 1e-3

class ExponentialScheduler(Scheduler):
    """Exponential lr scheduler"""

    config: ExponentialSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        gamma = (self.config.min_lr / self.config.init_lr) ** (1 / self.config.max_steps)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        return scheduler


@dataclass
class CustomExponentialLRSchedulerConfig(SchedulerConfig):
    """Config for custom exponential lr scheduler"""

    _target: Type = field(default_factory=lambda: CustomExponentialLRScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    decay_steps: int = 20000
    """The number of steps to decay the learning rate."""
    min_lr: float = 1e-5
    """The minimum learning rate."""
    init_lr: float = 1e-3
    """The initial learning rate."""

class _CustomExponentialLR(LRScheduler):
    def __init__(self, optimizer, decay_steps, lr_init, eta_min=1e-5, last_epoch=-1, verbose=False):
        self.decay_steps = decay_steps
        self.eta_min = eta_min
        
        base_lr = lr_init
        if self.eta_min >= base_lr:
            raise ValueError(f"eta_min ({self.eta_min}) must be smaller than the initial learning rate ({base_lr}).")
        if self.decay_steps <= 0:
            gamma = 1.0
        else:
            gamma = (self.eta_min / base_lr) ** (1 / self.decay_steps)
        
        self.gamma_per_group = [gamma for _ in optimizer.param_groups]

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch >= self.decay_steps:
            return [self.eta_min for _ in self.base_lrs]
        
        return [base_lr * (gamma ** self.last_epoch) 
                for base_lr, gamma in zip(self.base_lrs, self.gamma_per_group)]


class CustomExponentialLRScheduler(Scheduler):
    """Custom exponential lr scheduler"""

    config: CustomExponentialLRSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = _CustomExponentialLR(optimizer=optimizer,
                                         decay_steps=self.config.decay_steps,
                                         eta_min=self.config.min_lr,
                                         lr_init=self.config.init_lr)
        return scheduler

