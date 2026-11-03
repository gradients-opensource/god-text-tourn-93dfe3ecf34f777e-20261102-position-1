
import random
import time
import zlib
from collections import defaultdict
from typing import Callable, Optional

import numpy as np

import host_cpu


_SHINGLE_K = 5
_NUM_HASHES = 64
_NUM_BANDS = 16
_JACCARD_THRESH = 0.8
_MAX_BUCKET_SIZE = 200

_PRIME = 2_147_483_647
_rng_coeff = random.Random(42)
_HASH_A = np.array(
    [_rng_coeff.randint(1, _PRIME - 1) for _ in range(_NUM_HASHES)],
    dtype=np.int64,
)
_HASH_B = np.array(
    [_rng_coeff.randint(0, _PRIME - 1) for _ in range(_NUM_HASHES)],
    dtype=np.int64,
)


def concat_all_text(item: dict) -> str:
    parts = []
    for v in item.values():
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def _shingle_hashes(text: str) -> np.ndarray:
    k = _SHINGLE_K
    if len(text) < k:
        return np.array([zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF], dtype=np.int64)
    hashes = set()
    for i in range(len(text) - k + 1):
        hashes.add(zlib.crc32(text[i : i + k].encode("utf-8")) & 0x7FFFFFFF)
    return np.array(list(hashes), dtype=np.int64)


def _minhash_sig(shingles: np.ndarray) -> np.ndarray:
    s = shingles.reshape(1, -1)
    vals = (_HASH_A.reshape(-1, 1) * s + _HASH_B.reshape(-1, 1)) % _PRIME
    return vals.min(axis=1)


def _sigs_of_texts(texts: list[str]) -> list[np.ndarray]:
    return [_minhash_sig(_shingle_hashes(t)) for t in texts]


POOL_MIN_ROWS = 2000


def _signatures(items: list[dict], text_fn: Callable[[dict], str]) -> list[np.ndarray]:
    texts = [text_fn(item) for item in items]
    workers = host_cpu.current().tokenize_workers
    parts, _ = host_cpu.pool_map(_sigs_of_texts, host_cpu.chunk(texts, workers * 4),
                                   workers=workers, min_items=POOL_MIN_ROWS,
                                   n_items=len(texts))
    sigs = [s for part in parts for s in part]
    return sigs


def _near_dedup_groups(
    items: list[dict],
    text_fn: Callable[[dict], str],
) -> list[list[int]]:
    n = len(items)
    rows_per_band = _NUM_HASHES // _NUM_BANDS

    sigs = _signatures(items, text_fn)

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(_NUM_BANDS):
            start = b * rows_per_band
            key = (b, tuple(sig[start : start + rows_per_band].tolist()))
            buckets[key].append(idx)

    candidates = set()
    for indices in buckets.values():
        if 1 < len(indices) <= _MAX_BUCKET_SIZE:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    candidates.add(
                        (min(indices[i], indices[j]), max(indices[i], indices[j]))
                    )

    edges = []
    for i, j in candidates:
        matches = int(np.sum(sigs[i] == sigs[j]))
        if matches / _NUM_HASHES >= _JACCARD_THRESH:
            edges.append((i, j))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    group_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        group_map[find(idx)].append(idx)
    groups = list(group_map.values())

    n_dup_groups = sum(1 for g in groups if len(g) > 1)
    n_dup_items = sum(len(g) for g in groups if len(g) > 1)
    if n_dup_groups:
        print(
            f"Achei uns copycats... {n_dup_groups} grupos, "
            f"{n_dup_items}/{n} itens repetidos",
            flush=True,
        )
    else:
        print("Sem copycats, todo mundo é original", flush=True)

    return groups


def _quantile_bins(values: list[float], n_bins: int = 4) -> list[int]:
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: values[i])
    bins = [0] * n
    for rank, idx in enumerate(indexed):
        bins[idx] = min(rank * n_bins // n, n_bins - 1)
    return bins


def stratified_split(
    items: list[dict],
    dev_size: int,
    text_fn: Callable[[dict], str],
    difficulty_fn: Optional[Callable[[dict], float]] = None,
    seed: int = 42,
    near_dedup: bool = True,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    n = len(items)
    if n <= 1:
        return items[:0], items

    dev_size = max(1, min(dev_size, n - 1))

    if near_dedup and n > 10:
        groups = _near_dedup_groups(items, text_fn)
    else:
        groups = [[i] for i in range(n)]

    lengths: list[float] = []
    difficulties: list[float] = []
    for group in groups:
        rep = items[group[0]]
        text = text_fn(rep)
        lengths.append(float(len(text)))
        if difficulty_fn is not None:
            difficulties.append(float(difficulty_fn(rep)))
        else:
            difficulties.append(float(len(text)))

    n_groups = len(groups)
    n_strat_bins = 4
    len_bins = _quantile_bins(lengths, n_strat_bins)
    diff_bins = _quantile_bins(difficulties, n_strat_bins)

    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for gi in range(n_groups):
        strata[(len_bins[gi], diff_bins[gi])].append(gi)

    for key in strata:
        rng.shuffle(strata[key])

    total_items = sum(len(g) for g in groups)

    dev_group_set: set[int] = set()
    dev_count = 0

    targets: dict[tuple[int, int], int] = {}
    for key, gis in strata.items():
        stratum_size = sum(len(groups[gi]) for gi in gis)
        targets[key] = max(1, round(dev_size * stratum_size / total_items))

    for key, gis in strata.items():
        target = targets[key]
        taken = 0
        for gi in gis:
            if dev_count >= dev_size or taken >= target:
                break
            group_size = len(groups[gi])
            if dev_count + group_size > dev_size + max(5, dev_size // 10):
                continue
            dev_group_set.add(gi)
            taken += group_size
            dev_count += group_size

    if dev_count < dev_size:
        remaining = [gi for gi in range(n_groups) if gi not in dev_group_set]
        rng.shuffle(remaining)
        for gi in remaining:
            if dev_count >= dev_size:
                break
            dev_group_set.add(gi)
            dev_count += len(groups[gi])

    dev_idx: set[int] = set()
    for gi in dev_group_set:
        for idx in groups[gi]:
            dev_idx.add(idx)

    dev_items = [items[i] for i in range(n) if i in dev_idx]
    train_items = [items[i] for i in range(n) if i not in dev_idx]

    def _pcts(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        s = sorted(vals)
        n = len(s)
        return (
            f"p25={s[n // 4]:.0f} p50={s[n // 2]:.0f} "
            f"p75={s[3 * n // 4]:.0f} min={s[0]:.0f} max={s[-1]:.0f}"
        )

    dev_lens = [float(len(text_fn(d))) for d in dev_items]
    train_lens = [float(len(text_fn(t))) for t in train_items]
    dev_diffs = [float(difficulty_fn(d)) if difficulty_fn else float(len(text_fn(d))) for d in dev_items]
    train_diffs = [float(difficulty_fn(t)) if difficulty_fn else float(len(text_fn(t))) for t in train_items]

    dup_groups_in_dev = sum(1 for gi in dev_group_set if len(groups[gi]) > 1)
    dup_items_in_dev = sum(len(groups[gi]) for gi in dev_group_set if len(groups[gi]) > 1)

    dev_strata = set()
    for gi in dev_group_set:
        dev_strata.add((len_bins[gi], diff_bins[gi]))

    print(f"{len(dev_items)} pro teste, {len(train_items)} pro treino", flush=True)
    print(f"Escalação do teste:", flush=True)
    print(f"  Copycats no teste: {dup_groups_in_dev} ({dup_items_in_dev} itens)", flush=True)
    print(f"  Variedade: {len(dev_strata)}/{len(strata)} sabores cobertos", flush=True)
    print(f"  Tamanho:      teste  {_pcts(dev_lens)}", flush=True)
    print(f"                treino {_pcts(train_lens)}", flush=True)
    print(f"  Dificuldade:  teste  {_pcts(dev_diffs)}", flush=True)
    print(f"                treino {_pcts(train_diffs)}", flush=True)

    return dev_items, train_items
