# Case study: YOLO11n → FAD-Net

## Setting

- Baseline: Ultralytics YOLO11n, cotton flower detection, IID split
- Protocol (example): 250 epochs, batch 16, imgsz 640, SGD, seeds `[0,26,42]`, `pretrained=False`, `amp=False`, patience 60
- Gates vs baseline means: P > 82.19, mAP@50 > 84.37, params < 2,582,347
- Judge on **test** set, not val

## Final model

**FAD-Net（Faster–ADown–DySample Network）**

| Replace | With |
|---------|------|
| Neck `nn.Upsample` | `DySample` |
| Neck stride-2 `Conv` | `ADown` |
| Neck `C3k2` | `C3k2_FasterBlock` |

Backbone: left as-is in the winning run (drop-ins in backbone are allowed by the skill; full backbone swap is not).

| Model | P/% | mAP@50/% | Parameters | GFLOPs |
|-------|-----|----------|------------|--------|
| yolo11n | 82.19±1.54 | 84.37±0.44 | 2,582,347 | 6.44 |
| FAD-Net | 83.25±0.19 | 84.53±0.23 | 2,421,083 | 6.00 |

## Iteration log (why not stop earlier)

| Ver | Modules | Test P | Test mAP50 | Params↓ | Note |
|-----|---------|--------|------------|---------|------|
| v5 | FasterBlock + ADown + C2PSA_EMA | 80.89 | 84.11 | yes | Val strong, test P weak (EMA overfit risk) |
| v6 | DySample + GSConv + FasterBlock | 81.83 | 84.74✓ | yes | mAP ok, P −0.36 |
| v7 | DySample + ADown + CSPHet | 81.34 | 84.50✓ | yes | mAP ok, P short |
| **v8** | **DySample + ADown + FasterBlock** | **83.25✓** | **84.54✓** | **yes** | Pass → named FAD-Net |

## Practical tips that mattered

1. **Param check before train** — fail fast if not lighter.
2. **Test ≠ val** — several combos looked great on val and missed test P.
3. **Supervisor chain** — auto-start next yaml when gates fail; save versioned summaries.
4. **DONE marker** — don’t treat mid-run `best.pt` as finished seed.
5. **Register modules correctly** — `CSPHet_SimAM` / fused blocks must be in `base_modules` + `repeat_modules` or channels break.
6. **Merge tables** — insert improved row after baseline nano; display name can be FAD-Net while run dirs stay `*_v8_*`.
7. **Archive sources** — copy only the three modules’ HTML from `改进方法` into `FAD-Net/materials/{DySample,ADown,FasterBlock}/` (no weights/code).

## Minimal yaml shape (Neck-only)

```yaml
# backbone: identical to yolo11.yaml
head:
  - [-1, 1, DySample, [2]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2_FasterBlock, [512, False]]
  - [-1, 1, DySample, [2]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2_FasterBlock, [256, False]]
  - [-1, 1, ADown, [256]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2_FasterBlock, [512, False]]
  - [-1, 1, ADown, [512]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2_FasterBlock, [1024, True]]
  - [[16, 19, 22], 1, Detect, [nc]]
```
