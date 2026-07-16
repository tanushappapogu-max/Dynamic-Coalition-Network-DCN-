import random
import re
from typing import Optional

import torch
from torch.utils.data import Dataset


VOCAB = {
    '<pad>': 0, '<sos>': 1, '<eos>': 2, '-': 3,
    '0': 4, '1': 5, '2': 6, '3': 7, '4': 8,
    '5': 9, '6': 10, '7': 11, '8': 12, '9': 13,
    '+': 14, '*': 15, '/': 16, '(': 17, ')': 18,
}
IDX_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB['<pad>']
SOS_IDX = VOCAB['<sos>']
EOS_IDX = VOCAB['<eos>']

OPS = ['+', '-', '*', '/']


def _random_operand():
    return str(random.randint(1, 9))


def _generate_expr(num_ops: int, max_nesting: int, _depth: int = 0) -> str:
    if num_ops == 0:
        return _random_operand()

    if max_nesting > 0 and _depth < max_nesting and num_ops >= 2 and random.random() < 0.3:
        inner_ops = random.randint(1, min(num_ops - 1, 3))
        inner = _generate_expr(inner_ops, max_nesting, _depth + 1)
        remaining = num_ops - inner_ops - 1
        op = random.choice(OPS)
        rest = _generate_expr(remaining, max_nesting, _depth)
        if random.random() < 0.5:
            return f"({inner}){op}{rest}"
        else:
            return f"{rest}{op}({inner})"

    left_ops = random.randint(0, num_ops - 1)
    right_ops = num_ops - 1 - left_ops
    left = _generate_expr(left_ops, max_nesting, _depth)
    right = _generate_expr(right_ops, max_nesting, _depth)
    op = random.choice(OPS)
    return f"{left}{op}{right}"


def _safe_eval(expr: str) -> Optional[int]:
    try:
        result = eval(expr)  # noqa: S307 — only evaluating self-generated arithmetic
        if not isinstance(result, (int, float)):
            return None
        if isinstance(result, float) and result != int(result):
            return None
        result = int(result)
        if abs(result) > 999:
            return None
        return result
    except (ZeroDivisionError, SyntaxError, OverflowError):
        return None


def generate_expression(num_ops: int, max_nesting: int) -> Optional[tuple[str, int]]:
    for _ in range(50):
        expr = _generate_expr(num_ops, max_nesting)
        result = _safe_eval(expr)
        if result is not None:
            return expr, result
    return None


def generate_dataset(
    size: int,
    num_ops_range: tuple[int, int],
    max_nesting: int,
    seed: int = 42,
    exclude: Optional[set[str]] = None,
    balanced: bool = True,
) -> list[tuple[str, str]]:
    rng_state = random.getstate()
    random.seed(seed)

    exclude = exclude or set()
    seen = set()

    if not balanced:
        data = []
        while len(data) < size:
            num_ops = random.randint(num_ops_range[0], num_ops_range[1])
            result = generate_expression(num_ops, max_nesting)
            if result is None:
                continue
            expr, val = result
            if expr in seen or expr in exclude:
                continue
            seen.add(expr)
            data.append((expr, str(val)))
        random.setstate(rng_state)
        return data

    types = ['add_sub_only', 'mul_div_only', 'mixed_precedence', 'parenthesized']
    per_type = size // len(types)
    remainder = size - per_type * len(types)
    buckets = {t: [] for t in types}
    targets = {t: per_type + (1 if i < remainder else 0) for i, t in enumerate(types)}

    attempts = 0
    max_attempts = size * 50
    while any(len(buckets[t]) < targets[t] for t in types) and attempts < max_attempts:
        attempts += 1
        num_ops = random.randint(num_ops_range[0], num_ops_range[1])
        result = generate_expression(num_ops, max_nesting)
        if result is None:
            continue
        expr, val = result
        if expr in seen or expr in exclude:
            continue
        etype = classify_expression(expr)
        if len(buckets[etype]) < targets[etype]:
            seen.add(expr)
            buckets[etype].append((expr, str(val)))

    data = []
    for t in types:
        data.extend(buckets[t])
    random.shuffle(data)

    random.setstate(rng_state)
    return data


def tokenize_input(expr: str, max_len: int) -> list[int]:
    tokens = [SOS_IDX]
    for ch in expr:
        if ch in VOCAB:
            tokens.append(VOCAB[ch])
    tokens.append(EOS_IDX)
    while len(tokens) < max_len:
        tokens.append(PAD_IDX)
    return tokens[:max_len]


def tokenize_output(result_str: str, max_len: int) -> list[int]:
    tokens = []
    for ch in result_str:
        if ch in VOCAB:
            tokens.append(VOCAB[ch])
    tokens.append(EOS_IDX)
    while len(tokens) < max_len:
        tokens.append(PAD_IDX)
    return tokens[:max_len]


def decode_output(token_ids: list[int]) -> str:
    chars = []
    for idx in token_ids:
        if idx == EOS_IDX:
            break
        if idx == PAD_IDX or idx == SOS_IDX:
            continue
        chars.append(IDX_TO_TOKEN.get(idx, '?'))
    return ''.join(chars)


def classify_expression(expr: str) -> str:
    has_parens = '(' in expr
    ops_found = set()
    for ch in expr:
        if ch in {'+', '-', '*', '/'}:
            ops_found.add(ch)

    if has_parens:
        return 'parenthesized'
    if '*' in ops_found or '/' in ops_found:
        if '+' in ops_found or '-' in ops_found:
            return 'mixed_precedence'
        return 'mul_div_only'
    return 'add_sub_only'


class ArithmeticDataset(Dataset):
    def __init__(self, data: list[tuple[str, str]], max_input_len: int = 64, max_output_len: int = 5):
        self.data = data
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        expr, result = self.data[idx]
        input_ids = torch.tensor(tokenize_input(expr, self.max_input_len), dtype=torch.long)
        target_ids = torch.tensor(tokenize_output(result, self.max_output_len), dtype=torch.long)
        return input_ids, target_ids


def create_datasets(config: dict) -> dict[str, ArithmeticDataset]:
    dc = config['data']
    max_in = dc['max_input_len']
    max_out = dc['max_output_len']

    train_data = generate_dataset(
        dc['train_size'], tuple(dc['train_num_ops']), dc['train_max_nesting'], seed=dc['seed']
    )
    train_exprs = {e for e, _ in train_data}

    val_data = generate_dataset(
        dc['val_size'], tuple(dc['train_num_ops']), dc['train_max_nesting'],
        seed=dc['seed'] + 1, exclude=train_exprs,
    )
    val_exprs = {e for e, _ in val_data}

    test_data = generate_dataset(
        dc['test_size'], tuple(dc['train_num_ops']), dc['train_max_nesting'],
        seed=dc['seed'] + 2, exclude=train_exprs | val_exprs,
    )

    gen_data = generate_dataset(
        dc['gen_test_size'], tuple(dc['gen_num_ops']), dc['gen_max_nesting'],
        seed=dc['seed'] + 3,
    )

    return {
        'train': ArithmeticDataset(train_data, max_in, max_out),
        'val': ArithmeticDataset(val_data, max_in, max_out),
        'test': ArithmeticDataset(test_data, max_in, max_out),
        'gen_test': ArithmeticDataset(gen_data, max_in, max_out),
    }
