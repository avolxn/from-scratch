import torch


class ScalerInterface:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError


class StaticScaler:
    def __init__(self, scale: float = 1024.0):
        self._scale = scale

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self._scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        inverse_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(inverse_scale)
        optimizer.step()

    def update(self) -> None:
        pass


class DynamicScaler:
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self._scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self._scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self._found_inf = False
        params = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    params.append(param)

        for param in params:
            if not torch.isfinite(param.grad).all():
                self._found_inf = True
                break

        if not self._found_inf:
            inverse_scale = 1.0 / self._scale
            for param in params:
                param.grad.data.mul_(inverse_scale)
            optimizer.step()

    def update(self) -> None:
        if self._found_inf:
            self._scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale *= self.growth_factor
                self._growth_tracker = 0
