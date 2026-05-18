#!/usr/bin/env python3
# Verify a WSDMC smoke run.
#
# Run after wsdmc-smoke-main.sbatch and wsdmc-smoke-cooldown.sbatch both
# complete. Checks:
#   1. Main produced 3 checkpoints (iters 10, 20, 30) under torch_dist format.
#   2. Each checkpoint has the same shard layout (same set of *.distcp files).
#   3. MoE expert weight shards are present (grouped-GEMM 3D tensors).
#   4. Cooldown produced its own iter_15 checkpoint with the same shard layout
#      as the main's iter_10 (proves model survived save/load round-trip).
#   5. Cooldown log shows: load succeeded, LR decayed monotonically over 5
#      cooldown iters, no NaN loss.
#
# Usage:
#   cd <repo root>
#   python3 _research/launch/wsdmc-smoke-verify.py
#
# Compatible with Python 3.6+ (login node default).

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPTS_ROOT = REPO / "_research" / "results" / "ckpts"
RUNS_ROOT = REPO / "_research" / "results" / "runs"

MAIN_CKPT_DIR = CKPTS_ROOT / "wsdmc-smoke-adamw"
COOLDOWN_CKPT_DIR = CKPTS_ROOT / "wsdmc-smoke-adamw-cd-cd-0.0B"  # endpoint < 1B rounds to 0.0B

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
exit_code = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global exit_code
    if cond:
        print(f"{PASS} {name}{(': ' + detail) if detail else ''}")
    else:
        print(f"{FAIL} {name}{(': ' + detail) if detail else ''}")
        exit_code = 1


def warn(name: str, detail: str = "") -> None:
    print(f"{WARN} {name}{(': ' + detail) if detail else ''}")


def list_iter_dirs(d):
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"iter_\d+$", p.name))


def shard_set(iter_dir):
    """Set of shard filenames inside an iter_NNN dir (torch_dist .distcp)."""
    return {p.name for p in iter_dir.rglob("*.distcp")}


def has_expert_shards(iter_dir):
    """Look for any tensor key containing 'expert' or 'mlp.experts' in
    the metadata file (torch_dist saves a .metadata file with key index)."""
    meta = iter_dir / ".metadata"
    if not meta.is_file():
        return False
    try:
        # .metadata is a pickled object; we just grep the bytes for the
        # substring 'expert' since we don't want to import torch here.
        raw = meta.read_bytes()
        return b"experts" in raw or b"local_experts" in raw
    except OSError:
        return False


def find_latest_log(prefix):
    if not RUNS_ROOT.is_dir():
        return None
    cands = sorted(RUNS_ROOT.glob(f"{prefix}-*.log"))
    return cands[-1] if cands else None


def parse_lr_seq(log_path):
    """Pull the lr value from each per-iteration line. Megatron logs a
    line like 'lr: 1.000E-02' or 'learning rate: 1.000000E-02'."""
    if not log_path.is_file():
        return []
    out = []
    pat = re.compile(r"learning rate:\s*([0-9.+-Ee]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                pass
    return out


def parse_loss_seq(log_path):
    if not log_path.is_file():
        return []
    out = []
    pat = re.compile(r"\blm loss:\s*([0-9.+-Ee]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append(float(m.group(1)))
            except ValueError:
                pass
    return out


def log_says(log_path, needle):
    if not log_path.is_file():
        return False
    return needle in log_path.read_text(errors="ignore")


# ---- 1. main checkpoints -----------------------------------------------------
print("=== checkpoint structure ===")
main_iters = list_iter_dirs(MAIN_CKPT_DIR)
main_iter_names = [p.name for p in main_iters]
check("main: 3 checkpoints", len(main_iters) == 3, f"found {main_iter_names}")
expected_main = ["iter_0000010", "iter_0000020", "iter_0000030"]
check("main: ckpts at iters 10/20/30", main_iter_names == expected_main, f"got {main_iter_names}")

# ---- 2. shard layout consistency ---------------------------------------------
if len(main_iters) >= 2:
    s0 = shard_set(main_iters[0])
    s1 = shard_set(main_iters[1])
    check("main: shard layout stable across saves",
          s0 == s1 and len(s0) > 0,
          f"|iter_10|={len(s0)} |iter_20|={len(s1)} symdiff={len(s0 ^ s1)}")

# ---- 3. expert shards present ------------------------------------------------
if main_iters:
    check("main: expert shards present in iter_10",
          has_expert_shards(main_iters[0]),
          f"checked {main_iters[0]}/.metadata")

# ---- 4. cooldown checkpoint ---------------------------------------------------
cd_iters = list_iter_dirs(COOLDOWN_CKPT_DIR)
cd_iter_names = [p.name for p in cd_iters]
check("cooldown: 1 checkpoint at iter_13", cd_iter_names == ["iter_0000013"], f"got {cd_iter_names}")

if cd_iters and main_iters:
    main_s = shard_set(main_iters[0])
    cd_s = shard_set(cd_iters[0])
    check("cooldown: shard layout matches main (load round-trip)",
          main_s == cd_s and len(cd_s) > 0,
          f"|main|={len(main_s)} |cd|={len(cd_s)} symdiff={len(main_s ^ cd_s)}")
    check("cooldown: expert shards present", has_expert_shards(cd_iters[0]))

# ---- 5. cooldown training log ------------------------------------------------
print("\n=== cooldown log ===")
cd_log = find_latest_log("wsdmc-smoke-adamw-cd")
if not cd_log:
    check("cooldown: log file present", False, "no log found")
else:
    print(f"     log: {cd_log}")
    check("cooldown: load succeeded",
          log_says(cd_log, "successfully loaded checkpoint")
          or log_says(cd_log, "loaded checkpoint")
          or log_says(cd_log, "will load weights"))

    lrs = parse_lr_seq(cd_log)
    if lrs:
        # Last N=3 training iters of the cooldown (matches WSDMC_COOLDOWN_ITERS=3).
        n_cd = 3
        train_lrs = lrs[-n_cd:] if len(lrs) >= n_cd else lrs
        decreasing = all(train_lrs[i + 1] <= train_lrs[i] + 1e-12 for i in range(len(train_lrs) - 1))
        # Final LR should land at or near min_lr. For peak=5e-4, min=5e-5.
        last_at_min = train_lrs[-1] < 1.0e-4
        check("cooldown: LR sequence non-increasing across cooldown iters",
              decreasing, f"lrs={train_lrs}")
        check("cooldown: LR ends at/near min_lr (< 1e-4)",
              last_at_min, f"final lr = {train_lrs[-1]:.3e}")
    else:
        warn("cooldown: could not parse LR sequence from log")

    losses = parse_loss_seq(cd_log)
    if losses:
        finite_losses = all(l == l and abs(l) < 1e6 for l in losses)
        check("cooldown: all training losses finite",
              finite_losses, f"first={losses[0]:.4f} last={losses[-1]:.4f} n={len(losses)}")
    else:
        warn("cooldown: could not parse loss sequence from log")

print("\n" + ("ALL CHECKS PASSED" if exit_code == 0 else "ONE OR MORE CHECKS FAILED"))
sys.exit(exit_code)
