# Jasper's Experiment Skill

Cursor Agent Skill for YOLO module-replacement experiments (drop-in Neck/backbone modules, multi-seed gates, SCI-style `XXX-Net` naming, and archiving source tutorials).

## Install (Cursor)

Copy into your personal skills directory:

```bash
mkdir -p ~/.cursor/skills
cp -r yolo-module-improve ~/.cursor/skills/
```

Or clone this repo and symlink:

```bash
git clone <this-repo-url> ~/Jaspers-Experiment-Skill
ln -s ~/Jaspers-Experiment-Skill/yolo-module-improve ~/.cursor/skills/yolo-module-improve
```

## Contents

- `yolo-module-improve/SKILL.md` — main workflow
- `yolo-module-improve/reference.md` — FAD-Net case study

## What it does

1. Freeze baseline protocol and metrics  
2. Pick **3** drop-in modules from materials (no whole-backbone swap)  
3. Param ↓ + test-set P / mAP@50 ↑ gates  
4. Iterate until pass  
5. Name as **`XXX-Net`** from module initials  
6. Archive only the used tutorials from `改进方法` into `XXX-Net/materials/`
