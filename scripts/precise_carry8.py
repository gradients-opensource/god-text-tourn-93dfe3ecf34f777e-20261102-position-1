import math
from typing import Optional

import torch

BLOCK = 256
SPLIT = 8
EAGER_CHUNK = 1 << 22
MIN_8BIT = 4096
M32 = 0xFFFFFFFF


def dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8) -> torch.Tensor:
    data = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    data.append(0)
    data.append(1.0)
    assert len(data) == 2 ** total_bits
    for _ in range(256 - len(data)):
        data.append(0)
    data.sort()
    return torch.tensor(data, dtype=torch.float32)


try:
    from bitsandbytes.functional import create_dynamic_map
    QMAP_SOURCE = "bitsandbytes"
except Exception:
    create_dynamic_map = dynamic_map
    QMAP_SOURCE = "inline"
try:
    QMAP_S = create_dynamic_map(signed=True).to(torch.float32).contiguous()
    QMAP_U = create_dynamic_map(signed=False).to(torch.float32).contiguous()
except Exception:
    QMAP_S = dynamic_map(signed=True).contiguous()
    QMAP_U = dynamic_map(signed=False).contiguous()
    QMAP_SOURCE = "inline"
_QMAPS: dict = {}

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = tl = None
    HAS_TRITON = False


def qmaps(device) -> tuple:
    key = str(device)
    if key not in _QMAPS:
        _QMAPS[key] = (QMAP_S.to(device), QMAP_U.to(device))
    return _QMAPS[key]


def seed32(run_seed: int, step: int, tensor_idx: int) -> int:
    return (int(run_seed) * 0x9E3779B1 + int(step) * 0x85EBCA6B + int(tensor_idx) * 0xC2B2AE35
            + 0x27D4EB2F) & M32


def scalars(lr: float, b1: float, b2: float, eps: float, wd: float, step: int) -> dict:
    bc1 = 1.0 - b1 ** step
    c2 = math.sqrt(1.0 - b2 ** step)
    return {"b1": b1, "b2": b2, "omb1": 1.0 - b1, "omb2": 1.0 - b2, "eps_c2": eps * c2,
            "step_size": -lr * c2 / bc1, "decay": 1.0 - lr * wd}


def _f32(x: float, device) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float32, device=device)


def hash_uniform(seed: int, idx: torch.Tensor) -> torch.Tensor:
    x = (idx.to(torch.int64) ^ int(seed)) & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    x = x ^ (x >> 16)
    return (x >> 8).to(torch.float32) * (1.0 / 16777216.0)


def ulp_of(p16: torch.Tensor) -> tuple:
    e = (p16.view(torch.int16).to(torch.int32) & 0xFFFF) >> 7
    e = e & 0xFF
    tiny = e < 8
    ebits = torch.where(tiny, torch.ones_like(e), e - 7) << 23
    return ebits.view(torch.float32), tiny


def quantize_blocks(x: torch.Tensor, code: torch.Tensor) -> tuple:
    n = x.numel()
    pad = (-n) % BLOCK
    xf = torch.nn.functional.pad(x, (0, pad)) if pad else x
    blocks = xf.view(-1, BLOCK)
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    s = (blocks / absmax).clamp_(-1.0, 1.0).reshape(-1)
    hi = torch.searchsorted(code, s).clamp_(max=code.numel() - 1)
    lo = (hi - 1).clamp_(min=0)
    pick_lo = (s - code[lo]).abs() <= (s - code[hi]).abs()
    idx = torch.where(pick_lo, lo, hi).to(torch.uint8)
    return idx[:n], absmax.reshape(-1)


def dequantize_blocks(idx: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    n = idx.numel()
    pad = (-n) % BLOCK
    out = code[idx.to(torch.long)]
    if pad:
        out = torch.nn.functional.pad(out, (0, pad))
    out = (out.view(-1, BLOCK) * absmax.view(-1, 1)).reshape(-1)
    return out[:n]


@torch.no_grad()
def step_reference(p: torch.Tensor, g: torch.Tensor, state: dict, *, lr: float, b1: float,
                   b2: float, eps: float, wd: float, step: int, seed: int, mode: str,
                   fp32_moments: bool, chunk: int = EAGER_CHUNK) -> None:
    n = p.numel()
    pf, gf = p.view(-1), g.reshape(-1)
    dev = p.device
    sc = {k: _f32(v, dev) for k, v in scalars(lr, b1, b2, eps, wd, step).items()}
    qs, qu = qmaps(dev)
    chunk = max(BLOCK, (chunk // BLOCK) * BLOCK)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        blo, bhi = lo // BLOCK, (hi + BLOCK - 1) // BLOCK
        p16 = pf[lo:hi]
        g32 = gf[lo:hi].to(torch.float32)
        ok = torch.isfinite(g32)
        g32 = torch.where(ok, g32, torch.zeros_like(g32))
        if fp32_moments:
            m, v = state["exp_avg"][lo:hi], state["exp_avg_sq"][lo:hi]
        else:
            m = dequantize_blocks(state["m8"][lo:hi], state["absmax_m"][blo:bhi], qs)
            v = dequantize_blocks(state["v8"][lo:hi], state["absmax_v"][blo:bhi], qu)
        m_new = m * sc["b1"] + g32 * sc["omb1"]
        v_new = v * sc["b2"] + (g32 * g32) * sc["omb2"]
        upd = (m_new / (torch.sqrt(v_new) + sc["eps_c2"])) * sc["step_size"]
        if mode == "master":
            w = state["master"][lo:hi]
        elif mode == "carry":
            ulp, _tiny = ulp_of(p16)
            w = p16.to(torch.float32) + state["r8"][lo:hi].to(torch.float32) * (ulp * (1.0 / 256.0))
        else:
            w = p16.to(torch.float32)
        w = w * sc["decay"] + upd
        new = w.to(torch.bfloat16)
        new = torch.where(ok, new, p16)
        if fp32_moments:
            m_new = torch.where(ok, m_new, m)
            v_new = torch.where(ok, v_new, v)
        else:
            m_new = torch.where(ok, m_new, torch.zeros_like(m_new))
            v_new = torch.where(ok, v_new, torch.zeros_like(v_new))
        if mode == "master":
            state["master"][lo:hi] = torch.where(ok, w, state["master"][lo:hi])
        elif mode == "carry":
            c = w - new.to(torch.float32)
            ulp_n, tiny_n = ulp_of(new)
            q = (c * 256.0) / ulp_n
            u = hash_uniform(seed, torch.arange(lo, hi, device=dev))
            r_new = torch.floor(q + u).clamp_(-128.0, 127.0)
            r_new = torch.where(tiny_n, torch.zeros_like(r_new), r_new).to(torch.int8)
            state["r8"][lo:hi] = torch.where(ok, r_new, state["r8"][lo:hi])
        p16.copy_(new)
        if fp32_moments:
            state["exp_avg"][lo:hi] = m_new
            state["exp_avg_sq"][lo:hi] = v_new
        else:
            m8, am = quantize_blocks(m_new, qs)
            v8, av = quantize_blocks(v_new, qu)
            state["m8"][lo:hi] = m8
            state["v8"][lo:hi] = v8
            state["absmax_m"][blo:bhi] = am
            state["absmax_v"][blo:bhi] = av


@torch.no_grad()
def dequant_at(state: dict, idx: torch.Tensor, fp32_moments: bool, device) -> tuple:
    if fp32_moments:
        return state["exp_avg"][idx], state["exp_avg_sq"][idx]
    qs, qu = qmaps(device)
    blk = idx // BLOCK
    m = qs[state["m8"][idx].to(torch.long)] * state["absmax_m"][blk]
    v = qu[state["v8"][idx].to(torch.long)] * state["absmax_v"][blk]
    return m, v


@torch.no_grad()
def master_at(p: torch.Tensor, state: dict, idx: Optional[torch.Tensor], carry: bool) -> torch.Tensor:
    pf = p.view(-1)
    p16 = pf if idx is None else pf[idx]
    w = p16.to(torch.float32)
    if not carry:
        return w
    ulp, _tiny = ulp_of(p16)
    r = state["r8"] if idx is None else state["r8"][idx]
    return w + r.to(torch.float32) * (ulp * (1.0 / 256.0))


if HAS_TRITON:

    @triton.jit
    def _ulp_bf16(x16):
        e = ((x16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) >> 7) & 0xFF
        tiny = e < 8
        ebits = tl.where(tiny, 1, e - 7) << 23
        return ebits.to(tl.float32, bitcast=True), tiny

    @triton.jit
    def _to_bf16_rtn(w):
        b = w.to(tl.int32, bitcast=True)
        b = b + 0x7FFF + ((b >> 16) & 1)
        return (b >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _hash_uniform(seed, idx):
        x = (idx.to(tl.int64) ^ seed.to(tl.int64)) & 0xFFFFFFFF
        x = x ^ (x >> 16)
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x = x ^ (x >> 15)
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        x = x ^ (x >> 16)
        return (x >> 8).to(tl.float32) * 5.9604644775390625e-08

    @triton.jit
    def _quant(x, code_ptr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        xb = tl.reshape(x, (SPLIT, BLOCK))
        absmax = tl.max(tl.abs(xb), axis=1)
        absmax = tl.where(absmax == 0, 1.0, absmax)
        s = tl.clamp(tl.math.div_rn(xb, absmax[:, None]), -1.0, 1.0)
        lo = tl.zeros((SPLIT, BLOCK), dtype=tl.int32)
        hi = tl.full((SPLIT, BLOCK), 255, dtype=tl.int32)
        for _ in range(8):
            mid = (lo + hi) // 2
            val = tl.load(code_ptr + mid)
            higher = s > val
            lo = tl.where(higher, mid, lo)
            hi = tl.where(higher, hi, mid)
        d_lo = tl.abs(s - tl.load(code_ptr + lo))
        d_hi = tl.abs(s - tl.load(code_ptr + hi))
        idx = tl.where(d_lo <= d_hi, lo, hi)
        return tl.reshape(idx, (SPLIT * BLOCK,)), absmax

    @triton.jit(do_not_specialize=["n", "seed"])
    def carry8_kernel(p_ptr, g_ptr, m_ptr, v_ptr, am_ptr, av_ptr, r_ptr, qs_ptr, qu_ptr, n,
                      b1, b2, omb1, omb2, eps_c2, step_size, decay, seed,
                      BLOCK: tl.constexpr, SPLIT: tl.constexpr,
                      FP32_MOMENTS: tl.constexpr, CARRY: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * SPLIT * BLOCK + tl.arange(0, SPLIT * BLOCK)
        mask = offs < n
        blk = offs // BLOCK
        p16 = tl.load(p_ptr + offs, mask=mask, other=0.0)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ok = (g == g) & (tl.abs(g) < 3.4e38)
        g = tl.where(ok, g, 0.0)
        if FP32_MOMENTS:
            m = tl.load(m_ptr + offs, mask=mask, other=0.0)
            v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        else:
            m8 = tl.load(m_ptr + offs, mask=mask, other=0).to(tl.int32)
            v8 = tl.load(v_ptr + offs, mask=mask, other=0).to(tl.int32)
            am = tl.load(am_ptr + blk, mask=mask, other=0.0)
            av = tl.load(av_ptr + blk, mask=mask, other=0.0)
            m = tl.load(qs_ptr + m8) * am
            v = tl.load(qu_ptr + v8) * av
        m_new = m * b1 + g * omb1
        v_new = v * b2 + (g * g) * omb2
        upd = tl.math.div_rn(m_new, tl.math.sqrt_rn(v_new) + eps_c2) * step_size
        w = p16.to(tl.float32)
        if CARRY:
            r = tl.load(r_ptr + offs, mask=mask, other=0).to(tl.float32)
            ulp, _tiny = _ulp_bf16(p16)
            w = w + r * (ulp * (1.0 / 256.0))
        w = w * decay + upd
        new = _to_bf16_rtn(w)
        new = tl.where(ok, new, p16)
        if FP32_MOMENTS:
            m_new = tl.where(ok, m_new, m)
            v_new = tl.where(ok, v_new, v)
        else:
            m_new = tl.where(ok, m_new, 0.0)
            v_new = tl.where(ok, v_new, 0.0)
        tl.store(p_ptr + offs, new, mask=mask)
        if CARRY:
            c = w - new.to(tl.float32)
            ulp_n, tiny_n = _ulp_bf16(new)
            q = tl.math.div_rn(c * 256.0, ulp_n)
            u = _hash_uniform(seed, offs)
            r_new = tl.clamp(tl.floor(q + u), -128.0, 127.0)
            r_new = tl.where(tiny_n, 0.0, r_new)
            r_new = tl.where(ok, r_new, r)
            tl.store(r_ptr + offs, r_new.to(tl.int8), mask=mask)
        if FP32_MOMENTS:
            tl.store(m_ptr + offs, m_new, mask=mask)
            tl.store(v_ptr + offs, v_new, mask=mask)
        else:
            mq, am_new = _quant(m_new, qs_ptr, SPLIT, BLOCK)
            vq, av_new = _quant(v_new, qu_ptr, SPLIT, BLOCK)
            tl.store(m_ptr + offs, mq.to(tl.uint8), mask=mask)
            tl.store(v_ptr + offs, vq.to(tl.uint8), mask=mask)
            boffs = pid * SPLIT + tl.arange(0, SPLIT)
            bmask = boffs * BLOCK < n
            tl.store(am_ptr + boffs, am_new, mask=bmask)
            tl.store(av_ptr + boffs, av_new, mask=bmask)


@torch.no_grad()
def step_triton(p: torch.Tensor, g: torch.Tensor, state: dict, *, lr: float, b1: float,
                b2: float, eps: float, wd: float, step: int, seed: int, mode: str,
                fp32_moments: bool) -> None:
    if mode not in ("carry", "rtn"):
        raise ValueError(f"carry8 kernel: mode {mode!r}")
    n = p.numel()
    if n == 0:
        return
    pf = p.view(-1)
    gf = g.reshape(-1)
    if not gf.is_contiguous():
        gf = gf.contiguous()
    qs, qu = qmaps(p.device)
    sc = scalars(lr, b1, b2, eps, wd, step)
    carry = mode == "carry"
    dummy = state["r8"] if carry else pf
    if fp32_moments:
        m, v, am, av = state["exp_avg"], state["exp_avg_sq"], state["exp_avg"], state["exp_avg_sq"]
    else:
        m, v, am, av = state["m8"], state["v8"], state["absmax_m"], state["absmax_v"]
    grid = (triton.cdiv(n, SPLIT * BLOCK),)
    carry8_kernel[grid](pf, gf, m, v, am, av, dummy, qs, qu, n,
                        float(sc["b1"]), float(sc["b2"]), float(sc["omb1"]), float(sc["omb2"]),
                        float(sc["eps_c2"]), float(sc["step_size"]), float(sc["decay"]),
                        (int(seed) & M32) if carry else 0,
                        BLOCK=BLOCK, SPLIT=SPLIT, FP32_MOMENTS=bool(fp32_moments),
                        CARRY=carry, num_warps=4, enable_fp_fusion=False)


def init_state(p: torch.Tensor, state: dict, *, fp32_moments: bool, carry: bool,
               master: bool = False) -> None:
    n = p.numel()
    dev = p.device
    if fp32_moments:
        state["exp_avg"] = torch.zeros(n, dtype=torch.float32, device=dev)
        state["exp_avg_sq"] = torch.zeros(n, dtype=torch.float32, device=dev)
    else:
        blocks = (n + BLOCK - 1) // BLOCK
        state["m8"] = torch.zeros(n, dtype=torch.uint8, device=dev)
        state["v8"] = torch.zeros(n, dtype=torch.uint8, device=dev)
        state["absmax_m"] = torch.zeros(blocks, dtype=torch.float32, device=dev)
        state["absmax_v"] = torch.zeros(blocks, dtype=torch.float32, device=dev)
    if carry:
        state["r8"] = torch.zeros(n, dtype=torch.int8, device=dev)
    if master:
        state["master"] = p.detach().reshape(-1).to(torch.float32).clone()
    state["step"] = 0


def state_bytes(n: int, *, fp32_moments: bool, carry: bool, master: bool = False) -> int:
    per = 8.0 if fp32_moments else 2.0 + 8.0 / BLOCK
    per += 1.0 if carry else 0.0
    per += 4.0 if master else 0.0
    return int(per * n)


def _random_state(n: int, device, fp32_moments: bool, carry: bool, gen) -> dict:
    st: dict = {}
    if fp32_moments:
        st["exp_avg"] = (torch.randn(n, generator=gen) * 1e-3).to(device)
        st["exp_avg_sq"] = (torch.rand(n, generator=gen) * 1e-6).to(device)
    else:
        blocks = (n + BLOCK - 1) // BLOCK
        st["m8"] = torch.randint(0, 256, (n,), generator=gen).to(torch.uint8).to(device)
        st["v8"] = torch.randint(0, 256, (n,), generator=gen).to(torch.uint8).to(device)
        st["absmax_m"] = (torch.rand(blocks, generator=gen) * 1e-3).to(device)
        st["absmax_v"] = (torch.rand(blocks, generator=gen) * 1e-6).to(device)
    if carry:
        st["r8"] = torch.randint(-128, 128, (n,), generator=gen).to(torch.int8).to(device)
    st["step"] = 7
    return st


def selftest_cases(n: int = 5000, seed: int = 4242) -> list:
    gen = torch.Generator().manual_seed(seed)
    cases = []
    for mode in ("carry", "rtn"):
        for fp32 in (False, True):
            p0 = (torch.randn(n, generator=gen) * 0.05).to(torch.bfloat16)
            g0 = (torch.randn(n, generator=gen) * 1e-3).to(torch.bfloat16)
            g0[3] = float("nan")
            g0[777] = float("inf")
            p0[11] = 0.0
            st0 = _random_state(n, "cpu", fp32, mode == "carry", gen)
            hp = dict(lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, step=8, seed=seed32(1, 8, 3))
            cases.append((mode, fp32, p0, g0, st0, hp))
    return cases


def _clone_state(st: dict, device) -> dict:
    return {k: (v.clone().to(device) if torch.is_tensor(v) else v) for k, v in st.items()}


@torch.no_grad()
def selftest(device, n: int = 5000) -> dict:
    out = {"ok": True, "variants": {}, "error": None}
    if not HAS_TRITON or torch.device(device).type != "cuda":
        out["ok"] = False
        out["error"] = "no_triton" if not HAS_TRITON else "not_cuda"
        return out
    try:
        for mode, fp32, p0, g0, st0, hp in selftest_cases(n):
            pa, pb = p0.clone().to(device), p0.clone().to(device)
            sa, sb = _clone_state(st0, device), _clone_state(st0, device)
            step_reference(pa, g0.to(device), sa, mode=mode, fp32_moments=fp32, **hp)
            step_triton(pb, g0.to(device), sb, mode=mode, fp32_moments=fp32, **hp)
            torch.cuda.synchronize(device)
            same = {"p": bool(torch.equal(pa, pb))}
            for k in sa:
                if torch.is_tensor(sa[k]):
                    same[k] = bool(torch.equal(sa[k], sb[k]))
            out["variants"][f"{mode}/{'fp32' if fp32 else 'int8'}"] = same
            if not all(same.values()):
                out["ok"] = False
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return out
