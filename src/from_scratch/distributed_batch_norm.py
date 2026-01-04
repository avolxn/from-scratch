from typing import Any

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    Реализация синхронизированной батч-нормализации (SyncBN) как autograd-функции.

    Обеспечивает вычисление статистик (среднее и дисперсия) по всем процессам
    в распределенной группе (DDP), что позволяет использовать эффективный
    размер батча, равный сумме размеров батчей на всех GPU.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        running_mean: torch.Tensor | None,
        running_std: torch.Tensor | None,
        eps: float,
        momentum: float,
    ) -> torch.Tensor:
        """
        Прямой проход SyncBN.

        Args:
            ctx: Контекст для сохранения данных для обратного прохода.
            input (torch.Tensor): Входной тензор размерности (N, C, H, W).
            running_mean (torch.Tensor | None): Накопленное среднее (буфер слоя).
            running_std (torch.Tensor | None): Накопленное стандартное отклонение (буфер слоя).
            eps (float): Малое число для стабильности (избежание деления на ноль).
            momentum (float): Коэффициент инерции для обновления накопленных статистик.

        Returns:
            torch.Tensor: Нормализованный тензор той же размерности, что и вход.
        """
        N, C, H, W = input.shape
        dtype = input.dtype
        device = input.device

        local_count = torch.tensor([N * H * W], device=device, dtype=dtype)
        local_x_sum = torch.sum(input, dim=(0, 2, 3))
        local_x2_sum = torch.sum(input**2, dim=(0, 2, 3))

        vector = torch.cat([local_count, local_x_sum, local_x2_sum], dim=0)
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)

        global_count, global_x_sum, global_x2_sum = torch.split(vector, [1, C, C])

        mean = global_x_sum / global_count
        var = (global_x2_sum / global_count) - (mean**2)
        std = torch.sqrt(var + eps)

        if running_mean is not None and running_std is not None:
            running_mean.copy_((1 - momentum) * running_mean + momentum * mean)
            running_std.copy_((1 - momentum) * running_std + momentum * std)

        mean_4d = mean.view(1, C, 1, 1)
        std_4d = std.view(1, C, 1, 1)
        x_hat = (input - mean_4d) / std_4d

        ctx.save_for_backward(x_hat, std_4d, global_count)

        return x_hat

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        """
        Обратный проход SyncBN. Вычисляет градиент по входу с учетом зависимостей между GPU.

        Args:
            ctx: Контекст с сохраненными тензорами из forward.
            grad_output (torch.Tensor): Градиент функции потерь по выходу данного слоя.

        Returns:
            Tuple[torch.Tensor, None, None, None, None]: Градиент по входу (input).
                Остальные элементы None, так как по ним градиент не вычисляется.
        """
        x_hat, std, global_count = ctx.saved_tensors
        C = x_hat.shape[1]

        local_sum_grad_output = torch.sum(grad_output, dim=(0, 2, 3))
        local_sum_grad_output_x_hat = torch.sum(grad_output * x_hat, dim=(0, 2, 3))

        vector = torch.cat([local_sum_grad_output, local_sum_grad_output_x_hat], dim=0)
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)

        global_sum_grad_output, global_sum_grad_output_x_hat = vector.chunk(2)

        global_sum_grad_output = global_sum_grad_output.view(1, C, 1, 1)
        global_sum_grad_output_x_hat = global_sum_grad_output_x_hat.view(1, C, 1, 1)

        grad_input = (
            1.0
            / (global_count * std)
            * (global_count * grad_output - global_sum_grad_output - x_hat * global_sum_grad_output_x_hat)
        )

        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Модуль синхронизированной батч-нормализации без обучаемых параметров (gamma/beta).

    Args:
        num_features (int): Количество каналов во входном тензоре.
        eps (float): Значение для численной устойчивости. По умолчанию 1e-5.
        momentum (float): Коэффициент для обновления running stats. По умолчанию 0.1.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
        )
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_std", torch.ones((num_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Выполняет нормализацию входа.

        Args:
            input (torch.Tensor): Входной тензор (N, C, H, W).

        Returns:
            torch.Tensor: Результат нормализации.
        """
        if not self.training:
            C = self.running_mean.shape[0]
            mean = self.running_mean.view(1, C, 1, 1)
            std = self.running_std.view(1, C, 1, 1)
            return (input - mean) / std

        return sync_batch_norm.apply(
            input,
            self.running_mean,
            self.running_std,
            self.eps,
            self.momentum,
        )
