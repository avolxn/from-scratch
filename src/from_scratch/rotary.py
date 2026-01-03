import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even (dim % 2 == 0)")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._cache()

    def _cache(self):
        """Кэширует косинус и синус для всех позиций и измерений."""
        positions = torch.arange(self.max_seq_len, dtype=torch.float).unsqueeze(1)  # shape: [max_seq_len, 1]
        i = torch.arange(0, self.dim, 2).unsqueeze(0)
        thetas = 1.0 / (self.base ** (2 * i / self.dim))  # shape: [1, dim / 2]
        freqs = positions @ thetas  # shape: [max_seq_len, dim / 2]

        self.register_buffer("cos_cache", freqs.cos())
        self.register_buffer("sin_cache", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет Rotary Positional Embeddings к входному тензору x.

        Args:
            x (torch.Tensor):
            - shape: [batch_size, seq_len, num_heads, head_dim]

        Returns:
            x (torch.Tensor):
            - shape: [batch_size, seq_len, num_heads, head_dim]
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self.max_seq_len = self.max_seq_len * 2
            self._cache()

        # Добавляем измерение для num_heads
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(2)  # [batch_size, seq_len, 1, head_dim / 2]
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(2)  # [batch_size, seq_len, 1, head_dim / 2]

        # [x1, x2, x3, x4, ..., x(d-1), xd] -> [[x1, x2], [x3, x4], ..., [x(d-1), xd]]
        x_rotated = x.view(*x.shape[:-1], -1, 2)
        x_rotated[..., 0] = x_rotated[..., 0] * cos - x_rotated[..., 1] * sin
        x_rotated[..., 1] = x_rotated[..., 1] * cos + x_rotated[..., 0] * sin

        return x_rotated.flatten(-2)
