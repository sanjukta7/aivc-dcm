import torch
from abc import ABC, abstractmethod
from typing import Union

Tensor = torch.Tensor


class NoiseSchedule(ABC):
    @abstractmethod
    def rate(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def total(self, t: Tensor) -> Tensor:
        pass

    def __call__(self, t: Tensor) -> Tensor:
        return self.total(t)


class LogLinearNoise(NoiseSchedule):
    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def rate(self, t: Tensor) -> Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total(self, t: Tensor) -> Tensor:
        return -torch.log(1 - (1 - self.eps) * t)


class GeometricNoise(NoiseSchedule):
    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = torch.log(torch.tensor(sigma_max / sigma_min))

    def rate(self, t: Tensor) -> Tensor:
        return self.total(t) * self.log_ratio

    def total(self, t: Tensor) -> Tensor:
        return self.sigma_min ** (1 - t) * self.sigma_max ** t


def get_noise_schedule(name: str, **kwargs) -> NoiseSchedule:
    schedules = {
        "loglinear": LogLinearNoise,
        "geometric": GeometricNoise,
    }

    if name not in schedules:
        raise ValueError(f"Unknown noise schedule: {name}. Choose from {list(schedules.keys())}")

    return schedules[name](**kwargs)