---
name: yolo-module-improve
description: >-
  Improve Ultralytics YOLO detectors by drop-in module replacements in Neck
  and/or backbone (not whole-backbone swaps), keeping params lower and beating
  baseline P / mAP@50 on the same multi-seed protocol. Use when the user asks
  to improve YOLO11/YOLO, do module replacement (模块替换), Neck/backbone 轻量化,
  涨点, ablation-style improve iterations, or build named models like FAD-Net
  from 改进方法 materials.
---

# YOLO Module Improve (drop-in replace → multi-seed gate)

## Goal

From a frozen baseline (e.g. YOLO11n), produce an improved model that:

1. Uses **exactly three different modules** (from user materials when provided)
2. Has **fewer parameters** than baseline
3. Beats baseline **Precision (P)** and **mAP@50** on the **test** set (mean over seeds)
4. Uses **module drop-ins only** — backbone **may** be edited via replacements; do **not** replace the entire backbone with another network
5. Uses the **same train/eval protocol** as the baseline comparison
6. Leaves untouched modules’ args/structure alone; only change the replaced ones
7. Prefer diversity — do not fixate on one module family; iterate until gates pass

## Hard constraints

| Rule | Do | Don’t |
|------|----|-------|
| Scope | Drop-in replace modules in **Neck and/or backbone** (e.g. C3k2→Faster/CSPHet, Conv→ADown/GSConv, Upsample→DySample, C2PSA→attn variant) | Swap the **whole** backbone to MobileNet / Swin / RepViT / ShuffleNet / etc. |
| Topology | Keep stage layout / P3–P5 scales / Concat indices coherent | Redesign a new backbone DAG from scratch |
| Count | 3 distinct module types in the final story | Stack 5–10 unrelated tricks in one shot |
| Params | `n_params < baseline_params` before long training | Train a heavier model “hoping” accuracy saves it |
| Metrics | Judge on **test** mean over seeds (same as baseline tables) | Declare win from val curves alone |
| Protocol | Same `epochs/batch/imgsz/optimizer/seed/amp/pretrained/...` as baseline | Secret hyperparameter tuning only for the improve model |
| Wiring | Register modules in `tasks.py` `base_modules` / `repeat_modules` + parse hooks | Drop a class into yaml without parse support |

## Workflow

Copy and track:

```
Improve Progress:
- [ ] 1. Freeze baseline numbers (P, mAP50, params, protocol)
- [ ] 2. Inventory modules from 改进方法 / tutorials
- [ ] 3. Pick 3-module combo (Neck and/or backbone drop-ins)
- [ ] 4. Implement + register + yaml (no whole-backbone swap)
- [ ] 5. Param gate + smoke forward
- [ ] 6. Multi-seed train (resume-safe DONE flag)
- [ ] 7. Test-set eval → pass/fail gates
- [ ] 8. If fail: next combo (log why) → repeat from 3
- [ ] 9. Name model (`XXX-Net`), merge into baseline tables
- [ ] 10. Archive source tutorials from 改进方法 into `XXX-Net/materials/`
```

### 1) Freeze baseline

Read baseline summary/detail. Record:

- `TARGET_P`, `TARGET_MAP50`, `BASELINE_PARAMS`
- Train kwargs: data yaml, epochs, batch, imgsz, device, patience, optimizer, seeds, `pretrained`, `amp`, workers

Never change these for improve runs unless the user explicitly allows.

### 2) Inventory materials

From the user’s module folder (HTML/CSDN/code extracts):

- **Allowed**: Neck and backbone **module** tutorials (DySample, GSConv, ADown, CSPHet, FasterBlock→C3k2, SimAM/EMA on C2PSA, Ghost/PConv drop-ins, etc.)
- **Disallowed by default**: “将 backbone 替换为 MobileNet/Swin/…” style full swaps — extract reusable blocks only if they can be drop-ins inside the existing stages
- Mix Neck + backbone drop-ins freely within the 3-module budget

### 3) Choose a 3-module combo

Default strategy (worked for cotton / YOLO11n → FAD-Net; Neck-heavy is fine, not mandatory):

1. **Better upsample**: DySample (replace `nn.Upsample`)
2. **Lighter downsample**: ADown or GSConv (replace stride-2 `Conv` in Neck and/or backbone)
3. **Lighter block**: C3k2_FasterBlock or CSPHet (replace `C3k2` in Neck and/or backbone)

Keep yaml args for **unchanged** layers identical to baseline. Replaced slots may change module class; do not silently change args on layers you are not replacing.

**Selection heuristics from failed/passed runs:**

- Val ≫ test P → suspect heavy attn add-ons; try a different drop-in set (not “never touch backbone”)
- mAP50↑ but P short → keep the mAP-winning pair, swap the third module (or fuse param-free SimAM)
- Params not lower → drop param-heavy attentions; prefer PConv/HetConv/ADown/GSConv drop-ins

### 4) Implement modules

Typical Ultralytics touch points:

- `ultralytics/nn/extra_modules/*.py` — module code
- `extra_modules/__init__.py`, `nn/modules/__init__.py` — exports
- `nn/tasks.py` — imports; add to `base_modules` / `repeat_modules`; special parse for DySample/SimAM/EMA if needed
- Avoid circular imports (lazy-import fused attentions if needed)

Yaml: copy baseline `yolo11.yaml` scale `n`; edit **only** the entries you replace in `backbone` and/or `head`. Keep stage count and feature-map indices consistent.

### 5) Param gate + smoke

```python
from ultralytics import YOLO
m = YOLO(yaml_path)
n = sum(p.numel() for p in m.model.parameters())
assert n < BASELINE_PARAMS
```

Also run one dummy forward. Fix channel/`groups`/`p` divisibility before training.

### 6) Multi-seed train

Match baseline protocol. Practical resume rules:

- `is_done` = explicit `DONE` file after train finishes — **not** mere presence of `best.pt`
- After each seed: write `DONE`; then eval all seeds on `split=test`
- Log `console_*.log` + versioned `improve_vN_summary.json`
- Optional supervisor: on fail, launch next yaml (v6→v7→…)

### 7) Pass / fail

Pass iff **all** true on test means:

- `mean_P > TARGET_P`
- `mean_mAP50 > TARGET_MAP50`
- `mean_params < BASELINE_PARAMS`

On fail: keep a one-line postmortem (which gate failed, suspected cause), pick a **new** 3-module set, do not endlessly retune the same yaml.

### 8) Finalize — SCI Q1-style name (`XXX-Net`)

After gates pass, **name the model from the three modules** (not from “improve_v8”):

**Rules (SCI / Q1 detector papers):**

1. Format: **`XXX-Net`** only (ASCII letters + hyphen + `Net`). No `yolo11n_improve_*` in tables or paper text.
2. **`XXX`** = initials/abbreviations of the **three** drop-in modules, in a stable order:
   - Prefer **dataflow order** (e.g. upsample → block → downsample), or
   - The order used in the Methods contribution list
3. Provide a **full English expansion** once: `XXX-Net (Full–Module–Names Network)`.
4. Use en-dash `–` between module words in the expansion; keep short name with ASCII hyphen `-`.
5. Avoid marketing words, dataset names, or “Improved-YOLO” in the official model name.
6. Tables / figures / Methods use the short name; first mention in prose uses short + expansion.

**Example (this project):**

| Module | Token |
|--------|--------|
| C3k2_FasterBlock (FasterNet) | Faster → **F** |
| ADown | ADown → **A** |
| DySample | DySample → **D** |

→ **FAD-Net** (Faster–ADown–DySample Network)

Then:

- Merge into baseline summary/detail with `Model=FAD-Net`
- Keep weight dirs stable; rename **display** name only if paths are frozen
- Write `full_name` into improve summary JSON / RESULT.md

### 9) Archive module source files from `改进方法`

After naming, **copy only the original tutorials** used for the three modules into a standalone folder. Do **not** dump weights/code/results here unless the user asks.

**Layout:**

```
<project>/XXX-Net/
├── README.md                 # short name + which modules
└── materials/
    ├── <ModuleA>/            # original HTML/md from 改进方法
    ├── <ModuleB>/
    └── <ModuleC>/
```

**Rules:**

1. Source root is the user’s materials dir (e.g. `改进方法/`), including subfolders (`网页/`, `csdn/`, `sun777…/`, …).
2. Copy **only** files that correspond to the **three winning modules** (search by module keywords: DySample, ADown, FasterBlock, …).
3. Prefer original HTML/source; skip `/tmp` extracts, training logs, checkpoints, yaml, and ultralytics code unless requested.
4. Do **not** delete or move files out of `改进方法` — **copy**.
5. One short `README.md`: model full name, letter→module map, and that files are unmodified copies.

**Example (this project):** `/root/autodl-tmp/FAD-Net/materials/{DySample,ADown,FasterBlock}/`
## Train script skeleton

Required knobs (fill from baseline):

- `DATA`, `YAML`, `SEEDS`, `TARGET_*`, `BASELINE_PARAMS`
- `model.train(..., pretrained=False/True as baseline, amp=..., optimizer=..., patience=..., exist_ok=True, resume=...)`
- `model.val(..., split="test")` for reporting

## Anti-patterns

- Declaring success from mid-train val mAP
- Changing lr/aug/epochs only for the improve model
- Inserting extra layers that break Concat indices without updating them
- Registering a module in yaml but not in `base_modules` (silent wrong ctor args / zero channels)
- Using `best.pt` existence as “training finished”

## Case study

For the FAD-Net cotton / YOLO11n run (modules, fails, final numbers), see [reference.md](reference.md).
