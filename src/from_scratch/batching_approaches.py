import os

import numpy as np
import torch
from torch.utils.data import IterableDataset, Sampler
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

MAX_LENGTH = 640

_CACHED_DATA = None


def get_tokenizer() -> AutoTokenizer:
    """Initializes and returns the BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = 1e9
    return tokenizer


def load_data(
    data_path: str, tokenizer: AutoTokenizer, max_length: int, limit: int = 10000
) -> list[tuple[str, torch.Tensor]]:
    """Loads and tokenizes data with caching to avoid redundant processing."""
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    data = []
    if not os.path.exists(data_path):
        return []

    files = sorted([f for f in os.listdir(data_path) if f.startswith("train-") and f.endswith(".txt")])
    for file in files:
        with open(os.path.join(data_path, file), encoding="utf-8") as f:
            for line in f:
                if len(data) >= limit:
                    break
                line = line.strip()
                if not line or (line.startswith("=") and line.endswith("=")):
                    continue
                tokens = tokenizer.encode(line, add_special_tokens=False)[:max_length]
                data.append((line, torch.tensor(tokens, dtype=torch.long)))
            if len(data) >= limit:
                break

    _CACHED_DATA = data
    return data


class StaticPaddingDataset(Dataset):
    """Dataset that pads every sample to a fixed maximum length."""

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = get_tokenizer()
        self.max_length = max_length
        self.data = load_data(data_path, self.tokenizer, max_length)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        text, tokens = self.data[idx]
        padded = torch.zeros(self.max_length, dtype=torch.long)
        padded[: len(tokens)] = tokens
        return text, padded


class DynamicPaddingDataset(Dataset):
    """Dataset that returns raw tokens for dynamic padding in collate_fn."""

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = get_tokenizer()
        self.max_length = max_length
        self.data = load_data(data_path, self.tokenizer, max_length)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.data[idx]


class BucketDataset(Dataset):
    """Dataset designed to be used with BucketBatchSampler."""

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = get_tokenizer()
        self.max_length = max_length
        self.data = load_data(data_path, self.tokenizer, max_length)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.data[idx]


class PackedDataset(IterableDataset):
    """Dataset that packs multiple sequences into fixed-length blocks."""

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = get_tokenizer()
        self.max_length = max_length
        self.data = load_data(data_path, self.tokenizer, max_length)

    def __iter__(self):
        token_stream = []
        for i, (_, tokens) in enumerate(self.data):
            for token in tokens.tolist():
                token_stream.append((token, i))

        for i in range(0, len(token_stream) - self.max_length + 1, self.max_length):
            chunk = token_stream[i : i + self.max_length]
            chunk_tokens = [token for token, _ in chunk]
            chunk_seq_ids = [seq_id for _, seq_id in chunk]

            boundaries = []
            if not chunk_seq_ids:
                continue
            curr_seq_id = chunk_seq_ids[0]
            start_idx = 0
            for j in range(1, len(chunk_seq_ids)):
                if chunk_seq_ids[j] != curr_seq_id:
                    boundaries.append((start_idx, j))
                    start_idx = j
                    curr_seq_id = chunk_seq_ids[j]
            boundaries.append((start_idx, len(chunk_seq_ids)))

            yield torch.tensor(chunk_tokens, dtype=torch.long), boundaries


def collate_fn(
    batch: list[tuple[str, torch.Tensor]], max_length: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads a list of sequences and generates shifted targets."""
    tokens = [item[1] for item in batch]

    if max_length is not None:
        padded_samples = []
        for token in tokens:
            if len(token) < max_length:
                padded = torch.zeros(max_length, dtype=torch.long)
                padded[: len(token)] = token
                padded_samples.append(padded)
            else:
                padded_samples.append(token[:max_length])
        samples = torch.stack(padded_samples)
    else:
        m_len = max(len(token) for token in tokens)
        padded_samples = []
        for token in tokens:
            padded = torch.zeros(m_len, dtype=torch.long)
            padded[: len(token)] = token
            padded_samples.append(padded)
        samples = torch.stack(padded_samples)

    targets = torch.roll(samples, -1, dims=1)
    targets[:, -1] = 0

    return samples, targets


class BucketBatchSampler(Sampler):
    """Sampler that groups sequences by length to minimize padding."""

    def __init__(self, dataset: Dataset, batch_size: int, k: int):
        self.batch_size = batch_size
        self.k = k
        self.batches = []

        len_to_indices = {}
        for i in range(len(dataset)):
            _, tokens = dataset[i]
            length = len(tokens)
            if length not in len_to_indices:
                len_to_indices[length] = []
            len_to_indices[length].append(i)

        lengths = sorted(len_to_indices.keys())

        if not lengths:
            return

        all_indices_in_bins = []
        bin_indices = []
        bin_min_len = lengths[0]

        for length in lengths:
            if length <= bin_min_len + k:
                bin_indices.extend(len_to_indices[length])
            else:
                if bin_indices:
                    all_indices_in_bins.append(bin_indices)
                bin_indices = len_to_indices[length][:]
                bin_min_len = length
        if bin_indices:
            all_indices_in_bins.append(bin_indices)

        for bin_idxs in all_indices_in_bins:
            np.random.shuffle(bin_idxs)
            for i in range(0, len(bin_idxs), batch_size):
                self.batches.append(bin_idxs[i : i + batch_size])

        np.random.shuffle(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
