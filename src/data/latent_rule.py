"""Latent-rule task: four different algorithms, and WHICH one applies is
something you have to compute.

Input is a fixed-length digit string. The rule is  (sum of digits) mod 4:

    0  reverse     4 7 1 9 3 -> 3 9 1 7 4    (positional remap)
    1  sort        4 7 1 9 3 -> 1 3 4 7 9    (global comparison)
    2  increment   4 7 1 9 3 -> 5 8 2 0 4    (local, per-digit)
    3  prefix sum  4 7 1 9 3 -> 4 1 2 1 4    (sequential scan, mod 10)

WHY THIS TASK. Arithmetic could not answer the coalition question: every input
needed the same circuit, so specialised nodes had nothing to specialise in, and
our tsim metric was partly measuring honest overlap between op types that share
machinery. Here the four rules are genuinely different algorithms over a shared
substrate, so there is a GROUND-TRUTH PARTITION -- "did coalitions line up with
the four real jobs" is a fact, not an interpretation.

WHY THE RULE IS HIDDEN. A router that reads the raw input and predicts which
experts to use is very good when task identity is visible (a tag token, a
distinct vocabulary). Then routing wins by default and coalition structure adds
nothing. Here the selector is nonlinear in the input -- a LINEAR gate cannot
compute mod 4 -- so the input must be partly processed before you know what
machinery it needs. That is exactly the regime where selecting on a node's
actual response can beat predicting from the input.

Vocabulary is shared with src/data/arithmetic.py so the model stack is unchanged.
"""
import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.data.arithmetic import (
    VOCAB, IDX_TO_TOKEN, VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX,
)

SEQ_LEN = 5
N_RULES = 4
RULE_NAMES = ['reverse', 'sort', 'increment', 'prefixsum']


def rule_of(digits: list[int]) -> int:
    """The latent selector. Nonlinear in the input on purpose."""
    return sum(digits) % N_RULES


def apply_rule(digits: list[int], rule: int) -> list[int]:
    if rule == 0:
        return list(reversed(digits))
    if rule == 1:
        return sorted(digits)
    if rule == 2:
        return [(d + 1) % 10 for d in digits]
    out, run = [], 0
    for d in digits:
        run = (run + d) % 10
        out.append(run)
    return out


def classify_rule(digits: list[int]) -> str:
    """Ground-truth subtask label -- the partition arithmetic never had."""
    return RULE_NAMES[rule_of(digits)]


def generate_dataset(
    size: int,
    seq_len: int = SEQ_LEN,
    seed: int = 42,
    exclude: Optional[set[tuple]] = None,
    force_rule: Optional[int] = None,
) -> list[tuple[list[int], list[int]]]:
    """Exactly size/4 examples per rule, deduplicated, disjoint from `exclude`.

    force_rule pins every example to one algorithm, removing the latent selector.
    That is the control that separates two failure modes: if a model cannot learn
    even ONE rule in isolation, the bottleneck is the architecture (the output
    head reads a mean-pooled sequence), not the hidden selector.
    """
    rng_state = random.getstate()
    random.seed(seed)

    exclude = exclude or set()
    seen: set[tuple] = set()

    if force_rule is not None:
        data = []
        while len(data) < size:
            digits = [random.randint(0, 9) for _ in range(seq_len)]
            key = tuple(digits)
            if key in seen or key in exclude:
                continue
            seen.add(key)
            data.append((digits, apply_rule(digits, force_rule)))
        random.shuffle(data)
        random.setstate(rng_state)
        return data

    per_rule = size // N_RULES
    remainder = size - per_rule * N_RULES
    targets = [per_rule + (1 if i < remainder else 0) for i in range(N_RULES)]
    buckets: list[list] = [[] for _ in range(N_RULES)]

    attempts = 0
    max_attempts = size * 200
    while any(len(buckets[r]) < targets[r] for r in range(N_RULES)) and attempts < max_attempts:
        attempts += 1
        digits = [random.randint(0, 9) for _ in range(seq_len)]
        key = tuple(digits)
        if key in seen or key in exclude:
            continue
        r = rule_of(digits)
        if len(buckets[r]) < targets[r]:
            seen.add(key)
            buckets[r].append((digits, apply_rule(digits, r)))

    data = [item for b in buckets for item in b]
    random.shuffle(data)
    random.setstate(rng_state)
    return data


def tokenize_input(digits: list[int], max_len: int) -> list[int]:
    tokens = [SOS_IDX] + [VOCAB[str(d)] for d in digits] + [EOS_IDX]
    tokens += [PAD_IDX] * (max_len - len(tokens))
    return tokens[:max_len]


def tokenize_output(digits: list[int], max_len: int) -> list[int]:
    tokens = [VOCAB[str(d)] for d in digits] + [EOS_IDX]
    tokens += [PAD_IDX] * (max_len - len(tokens))
    return tokens[:max_len]


def decode_output(token_ids: list[int]) -> str:
    chars = []
    for idx in token_ids:
        if idx == EOS_IDX:
            break
        if idx in (PAD_IDX, SOS_IDX):
            continue
        chars.append(IDX_TO_TOKEN.get(int(idx), '?'))
    return ''.join(chars)


class LatentRuleDataset(Dataset):
    def __init__(self, data, max_input_len: int = 8, max_output_len: int = 6):
        self.data = data
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        digits, target = self.data[idx]
        x = torch.tensor(tokenize_input(digits, self.max_input_len), dtype=torch.long)
        y = torch.tensor(tokenize_output(target, self.max_output_len), dtype=torch.long)
        return x, y


def create_datasets(config: dict) -> dict[str, LatentRuleDataset]:
    dc = config['data']
    max_in = dc.get('max_input_len', 8)
    max_out = dc.get('max_output_len', 6)
    seq_len = dc.get('seq_len', SEQ_LEN)
    seed = dc.get('seed', 42)

    train = generate_dataset(dc['train_size'], seq_len, seed=seed)
    train_keys = {tuple(d) for d, _ in train}

    val = generate_dataset(dc['val_size'], seq_len, seed=seed + 1, exclude=train_keys)
    val_keys = {tuple(d) for d, _ in val}

    test = generate_dataset(dc['test_size'], seq_len, seed=seed + 2,
                            exclude=train_keys | val_keys)

    return {
        'train': LatentRuleDataset(train, max_in, max_out),
        'val': LatentRuleDataset(val, max_in, max_out),
        'test': LatentRuleDataset(test, max_in, max_out),
    }
