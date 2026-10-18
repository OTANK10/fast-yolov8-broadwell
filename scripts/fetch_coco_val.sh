#!/usr/bin/env bash
# COCO 2017 val images (5,000 jpgs, ~0.8 GB) onto scratch, for batch-N dumps:
#   bash scripts/fetch_coco_val.sh
#   python scripts/dump_layers.py --batch 16 --images /scratch/$USER/645/coco/val2017 --only 3x3s1
set -euo pipefail
DST=${SCR:-/scratch/$USER/645}/coco
mkdir -p "$DST" && cd "$DST"
[ -f val2017.zip ] || wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
[ -d val2017 ] || unzip -q val2017.zip
ls val2017 | wc -l
