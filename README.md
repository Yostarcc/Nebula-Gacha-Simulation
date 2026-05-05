# Nebula-Gacha-Simulation-Script(Gpt-Codex-5.3)
Nebula-Gacha-Simulation

## Environment
- python 3.13
- numpy

# Script
- Please change the `ROOT` with the real `*.json` files location

```
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\Workspace\Nebula")

# ---------- Load config ----------
with open(ROOT / "resources" / "bin" / "Gacha.json", "r", encoding="utf-8") as f:
    gacha = json.load(f)
with open(ROOT / "resources" / "bin" / "GachaStorage.json", "r", encoding="utf-8") as f:
    storage = json.load(f)
with open(ROOT / "resources" / "bin" / "GachaATypeProb.json", "r", encoding="utf-8") as f:
    aprobs = json.load(f)

# Use traveler limited pool example in your description
banner = gacha["10145"]
st = storage[str(banner["StorageId"])]

A_GROUP = int(st["ATypeGroup"])
A_UP_GUARANTEE = int(st.get("AUpGuaranteeTimes", 0))
A_UP_PROB = int(st.get("ATypeUpProb", 0))
B_TYPE_PROB = int(st.get("BTypeProb", 0))
B_GUARANTEE_TIMES = int(st.get("BGuaranteeTimes", 0)) or 10

# Build A-type probability table for this group
prob_map = {}
max_prob = 0
for obj in aprobs.values():
    if int(obj.get("Group", -1)) != A_GROUP:
        continue
    t = int(obj.get("Times", 0))
    p = int(obj.get("Prob", 200))
    prob_map[t] = p
    max_prob = max(max_prob, p)

DEFAULT_A = 200
ROLL_BASE = max(10000, max_prob)

max_t = max(prob_map.keys()) if prob_map else 0
prob_table = np.full(max_t + 1, prob_map.get(0, DEFAULT_A), dtype=np.int32)
for t, p in prob_map.items():
    if t >= len(prob_table):
        prob_table = np.pad(prob_table, (0, t - len(prob_table) + 1), mode="edge")
    prob_table[t] = p

# ---------- Simulation params ----------
TOTAL_PULLS = 100_000_000
LANES = 100_000
STEPS = TOTAL_PULLS // LANES
if STEPS * LANES != TOTAL_PULLS:
    raise ValueError("TOTAL_PULLS must be divisible by LANES")

rng = np.random.default_rng(20260505)

# pity states per lane
missA = np.zeros(LANES, dtype=np.int16)
missUpA = np.zeros(LANES, dtype=np.int16)
missB = np.zeros(LANES, dtype=np.int16)
bDebt = np.zeros(LANES, dtype=np.bool_)

# counters
five_cnt = 0
up_cnt = 0
b_or_higher_cnt = 0

# 10th-pity-trigger distribution
b_guard_pulls = 0
b_guard_a = 0
b_guard_b = 0

for _ in range(STEPS):
    # chanceA from curve, with A hard pity
    idx = np.minimum(missA.astype(np.int32), len(prob_table) - 1)
    chanceA = prob_table[idx].copy()

    hardA = missA >= (A_UP_GUARANTEE - 1)
    chanceA[hardA] = ROLL_BASE

    # UP hard pity (forces A_UP package)
    forceAUp = missUpA >= (A_UP_GUARANTEE - 1)

    # B pity trigger
    bTrig = missB >= (B_GUARANTEE_TIMES - 1)
    chanceB = np.where(bTrig, ROLL_BASE, B_TYPE_PROB).astype(np.int32)

    # roll
    r = rng.integers(1, ROLL_BASE + 1, size=LANES, dtype=np.int32)

    isA = forceAUp | (r <= chanceA)
    isB = (~isA) & (r <= chanceB)
    isC = (~isA) & (~isB)

    # split A vs A_UP
    isAUp = np.zeros(LANES, dtype=np.bool_)
    isAUp[forceAUp] = True

    normalA = isA & (~forceAUp)
    cA = int(normalA.sum())
    if cA > 0:
        isAUp[normalA] = rng.integers(1, 10001, size=cA) <= A_UP_PROB

    # update pity counters
    newMissA = missA + 1
    newMissB = missB + 1
    newMissUpA = missUpA + 1
    newDebt = bDebt.copy()

    # A branch
    newMissA[isA] = 0
    newDebt[isA & bTrig] = True  # B pity consumed by A -> debt

    # B branch
    newMissB[isB] = 0
    newDebt[isB] = False

    # debt compensation: C -> B
    compensated = newDebt & isC
    finalB = isB | compensated
    newMissB[compensated] = 0
    newDebt[compensated] = False

    # UP pity counter
    newMissUpA[isAUp] = 0

    # stats
    a_n = int(isA.sum())
    b_n = int(finalB.sum())

    five_cnt += a_n
    up_cnt += int(isAUp.sum())
    b_or_higher_cnt += (a_n + b_n)

    # 10th-pity-trigger composition
    bg = int(bTrig.sum())
    if bg > 0:
        b_guard_pulls += bg
        b_guard_a += int((isA & bTrig).sum())
        b_guard_b += int((finalB & bTrig).sum())

    missA = newMissA
    missB = newMissB
    missUpA = newMissUpA
    bDebt = newDebt

# ---------- Results ----------
p5 = five_cnt / TOTAL_PULLS * 100
pup = up_cnt / TOTAL_PULLS * 100
pbh = b_or_higher_cnt / TOTAL_PULLS * 100
pbg_a = b_guard_a / b_guard_pulls * 100
pbg_b = b_guard_b / b_guard_pulls * 100

print("CONFIG")
print("A_GROUP =", A_GROUP)
print("ROLL_BASE =", ROLL_BASE)
print("A_UP_GUARANTEE =", A_UP_GUARANTEE)
print("A_UP_PROB =", A_UP_PROB)
print("B_TYPE_PROB =", B_TYPE_PROB)
print("B_GUARANTEE_TIMES =", B_GUARANTEE_TIMES)
print()

print("SIM_RESULT (100,000,000 pulls)")
print(f"5★ overall: {p5:.6f}%")
print(f"UP 5★ overall: {pup:.6f}%")
print(f"4★+ overall: {pbh:.6f}%")
print(f"10th-pity-trigger pull -> 5★: {pbg_a:.6f}%")
print(f"10th-pity-trigger pull -> 4★: {pbg_b:.6f}%")
print()

print("OFFICIAL_DELTA")
print(f"5★ delta vs 2.247%: {p5 - 2.247:+.6f}%")
print(f"UP 5★ delta vs 1.250%: {pup - 1.250:+.6f}%")
print(f"4★+ delta vs 14.600%: {pbh - 14.600:+.6f}%")
print(f\"10th-pity 5★ delta vs 2.000%: {pbg_a - 2.000:+.6f}%\")
print(f\"10th-pity 4★ delta vs 98.000%: {pbg_b - 98.000:+.6f}%\")

```
