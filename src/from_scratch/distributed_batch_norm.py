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
        running_var: torch.Tensor | None,
        eps: float,
        momentum: float,
    ) -> torch.Tensor:
        """
        Прямой проход SyncBN.

        Args:
            ctx: Контекст для сохранения данных для обратного прохода.
            input (torch.Tensor): Входной тензор.
            running_mean (torch.Tensor | None): Накопленное среднее.
            running_var (torch.Tensor | None): Накопленная дисперсия.
            eps (float): Малое число для стабильности.
            momentum (float): Коэффициент инерции.

        Returns:
            torch.Tensor: Нормализованный тензор.
        """
        C = input.shape[1]
        dims = [0] + list(range(2, input.ndim))
        view_shape = [1, C] + [1] * (input.ndim - 2)

        local_count = torch.tensor([input.numel() // C], device=input.device, dtype=input.dtype)
        local_x_sum = torch.sum(input, dim=dims)
        local_x2_sum = torch.sum(input**2, dim=dims)

        vector = torch.cat([local_count, local_x_sum, local_x2_sum], dim=0)
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)

        global_count, global_x_sum, global_x2_sum = torch.split(vector, [1, C, C])

        mean = global_x_sum / global_count
        var = (global_x2_sum / global_count) - (mean**2)
        std = torch.sqrt(var + eps)

        if running_mean is not None and running_var is not None:
            unbiased_var = var * (global_count / (global_count - 1))
            running_mean.copy_((1 - momentum) * running_mean + momentum * mean)
            running_var.copy_((1 - momentum) * running_var + momentum * unbiased_var)

        mean_v = mean.view(view_shape)
        std_v = std.view(view_shape)
        x_hat = (input - mean_v) / std_v

        ctx.save_for_backward(x_hat, std_v, global_count)

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
        """
        x_hat, std, global_count = ctx.saved_tensors
        C = x_hat.shape[1]
        dims = [0] + list(range(2, x_hat.ndim))
        view_shape = [1, C] + [1] * (x_hat.ndim - 2)

        local_sum_grad_output = torch.sum(grad_output, dim=dims)
        local_sum_grad_output_x_hat = torch.sum(grad_output * x_hat, dim=dims)

        vector = torch.cat([local_sum_grad_output, local_sum_grad_output_x_hat], dim=0)
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)

        global_sum_grad_output, global_sum_grad_output_x_hat = vector.chunk(2)

        global_sum_grad_output = global_sum_grad_output.view(view_shape)
        global_sum_grad_output_x_hat = global_sum_grad_output_x_hat.view(view_shape)

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
        self.register_buffer("running_var", torch.ones((num_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Выполняет нормализацию входа.

        Args:
            input (torch.Tensor): Входной тензор.

        Returns:
            torch.Tensor: Результат нормализации.
        """
        if not self.training:
            C = self.running_mean.shape[0]
            view_shape = [1, C] + [1] * (input.ndim - 2)
            mean = self.running_mean.view(view_shape)
            var = self.running_var.view(view_shape)
            return (input - mean) / torch.sqrt(var + self.eps)

        return sync_batch_norm.apply(
            input,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
        )
