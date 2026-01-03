import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import regex as re

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


class TokenizerInterface(ABC):
    """Интерфейс для BPE токенизаторов."""

    special_tokens: dict[str, int]

    @abstractmethod
    def train(
        self,
        text_iterator: Iterator[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Обучает токенизатор на итераторе текстов."""
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Кодирует текст в список ID токенов."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Декодирует список ID токенов обратно в строку."""
        ...

    @abstractmethod
    def save(self, model_folder_path: str) -> None:
        """Сохраняет токенизатор в указанную папку."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, model_path: str) -> "TokenizerInterface":
        """Загружает токенизатор из файла."""
        ...

    def render_conversation(
        self,
        conversation: dict,
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """Токенизирует диалог для instruction-tuning.

        Args:
            conversation: Словарь с ключом "messages".
            max_tokens: Максимальное количество токенов.

        Returns:
            Кортеж (ids, mask), где mask=1 для токенов assistant.
        """
        ids: list[int] = []
        mask: list[int] = []

        messages = conversation.get("messages", [])
        if not messages:
            return ids, mask

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            nonlocal ids, mask
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        add_tokens(self.special_tokens["<|bos|>"], 0)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                add_tokens(self.special_tokens["<|system_start|>"], 0)
                add_tokens(self.encode(content), 0)
                add_tokens(self.special_tokens["<|system_end|>"], 0)

            elif role == "user":
                add_tokens(self.special_tokens["<|user_start|>"], 0)
                add_tokens(self.encode(content), 0)
                add_tokens(self.special_tokens["<|user_end|>"], 0)

            elif role == "assistant":
                add_tokens(self.special_tokens["<|assistant_start|>"], 0)

                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(self.special_tokens["<|python_start|>"], 1)
                            add_tokens(value_ids, 1)
                            add_tokens(self.special_tokens["<|python_end|>"], 1)
                        elif part["type"] == "python_output":
                            add_tokens(self.special_tokens["<|output_start|>"], 0)
                            add_tokens(value_ids, 0)
                            add_tokens(self.special_tokens["<|output_end|>"], 0)

                add_tokens(self.special_tokens["<|assistant_end|>"], 1)

        return ids[:max_tokens], mask[:max_tokens]


class Tokenizer(TokenizerInterface):
    """Byte-level BPE tokenizer with regex pre-tokenization"""

    def __init__(self, split_pattern: str | None = None):
        """Инициализирует токенизатор.

        Args:
            split_pattern: Шаблон регулярного выражения для предварительного разбиения текста.
        """
        self.split_pattern: re.Pattern = re.compile(split_pattern or GPT4_SPLIT_PATTERN)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: dict[str, int] = {}

    def _build_vocab(self) -> dict[int, bytes]:
        """Собирает словарь токенов на основе выполненных объединений и спецтокенов.

        Returns:
            dict[int, bytes]: Словарь, где ключ — ID токена, значение — его байты.
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (pair0, pair1), idx in self.merges.items():
            vocab[idx] = vocab[pair0] + vocab[pair1]

        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")

        return vocab

    def _merge(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """Заменяет все вхождения пары токенов на новый токен.

        Args:
            ids (list[int]): Список ID токенов.
            pair (tuple[int, int]): Пара токенов для объединения.
            idx (int): Новый ID токена.

        Returns:
            list[int]: Новый список ID после объединения.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _get_stats(
        self,
        ids: list[int],
        stats: dict[tuple[int, int], int] | None = None,
    ) -> dict[tuple[int, int], int]:
        """Подсчитывает частоту встречаемости пар последовательных токенов.

        Args:
            ids (list[int]): Список ID токенов.
            stats (dict[tuple[int, int], int] | None): Существующий словарь частот для обновления.

        Returns:
            dict[tuple[int, int], int]: Обновленный словарь частот пар токенов.
        """
        stats = stats if stats is not None else {}

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            stats[pair] = stats.get(pair, 0) + 1

        return stats

    def _register_special_tokens(self, special_tokens: list[str]) -> None:
        """Регистрирует специальные токены и назначает им ID.

        Args:
            special_tokens: Список строк для регистрации.
        """
        idx = 256 + len(self.merges)
        for special_token in special_tokens:
            if special_token not in self.special_tokens:
                self.special_tokens[special_token] = idx
                idx += 1

        self.vocab = self._build_vocab()

    def train(
        self,
        text_iterator: Iterator[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Обучает токенизатор на итераторе текстов.

        Args:
            text_iterator: Итератор строк для обучения.
            vocab_size: Целевой размер словаря.
            special_tokens: Список специальных токенов (по умолчанию SPECIAL_TOKENS).

        Raises:
            ValueError: Если vocab_size слишком мал.
        """
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS

        num_special = len(special_tokens)
        num_merges = vocab_size - 256 - num_special
        if num_merges < 0:
            raise ValueError(f"Размер словаря должен быть не менее {256 + num_special}")

        all_ids: list[list[int]] = []
        for text in text_iterator:
            chunks = re.findall(self.split_pattern, text)
            for chunk in chunks:
                all_ids.append(list(chunk.encode("utf-8")))

        merges = {}
        for _ in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for chunk_ids in all_ids:
                self._get_stats(chunk_ids, stats)

            if not stats:
                break

            best_pair = max(stats, key=lambda pair: stats[pair])
            idx = 256 + len(merges)
            all_ids = [self._merge(chunk_ids, best_pair, idx) for chunk_ids in all_ids]
            merges[best_pair] = idx

        self.merges = merges
        self._register_special_tokens(special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Кодирует отдельный чанк текста (без спецтокенов) в список ID.

        Args:
            chunk (str): Фрагмент текста.

        Returns:
            list[int]: Список ID токенов.
        """
        ids = list(chunk.encode("utf-8"))

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)

        return ids

    def encode(self, text: str) -> list[int]:
        """Кодирует текст в список ID токенов, включая обработку специальных токенов.

        Args:
            text (str): Входной текст.

        Returns:
            list[int]: Список ID токенов.
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                chunks = re.findall(self.split_pattern, part)
                for chunk in chunks:
                    ids.extend(self._encode_chunk(chunk))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Декодирует список ID токенов обратно в строку.

        Args:
            ids (list[int]): Список ID токенов.

        Returns:
            str: Декодированная строка (с заменой некорректных байтов).
        """
        tokens = [self.vocab[idx] for idx in ids]
        full_bytes = b"".join(tokens)

        return full_bytes.decode("utf-8", errors="replace")

    def save(self, model_folder_path: str) -> None:
        """Сохраняет модель токенизатора в указанную папку в формате JSON.

        Args:
            model_folder_path: Путь к папке, где будет создан файл tokenizer_bpe.json.
        """
        os.makedirs(model_folder_path, exist_ok=True)

        data = {
            "split_pattern": self.split_pattern.pattern,
            "special_tokens": self.special_tokens,
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
        }

        model_path = os.path.join(model_folder_path, "tokenizer_bpe.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_path: str) -> "Tokenizer":
        """Загружает токенизатор из файла JSON.

        Args:
            model_path: Путь к файлу tokenizer_bpe.json или папке с ним.

        Returns:
            Загруженный экземпляр Tokenizer.
        """
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "tokenizer_bpe.json")

        with open(model_path, encoding="utf-8") as f:
            data = json.load(f)

        instance = cls()
        instance.split_pattern = re.compile(data["split_pattern"])
        instance.special_tokens = data["special_tokens"]
        instance.merges = {(m[0], m[1]): m[2] for m in data["merges"]}
        instance.vocab = instance._build_vocab()
        return instance
