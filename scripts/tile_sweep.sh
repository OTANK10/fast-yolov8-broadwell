#!/usr/bin/env bash
# "How many accumulators does the microkernel need?"  Runs every s1 tile shape,
# single core, on one representative layer per channel count.
#   scripts/tile_sweep.sh            -> results/s1_tile_sweep.csv + a table on screen
# Expectation from the FMA numbers (5-cycle latency x 2 ports): FLOPs/cycle grows
# roughly linearly with the number of accumulators until 10, then flattens.
set -euo pipefail
reps=${REPS:-10}
layers=(model.2.m.0.cv1 model.4.m.0.cv1 model.6.m.0.cv1 model.8.m.0.cv1)   # C = 16, 32, 64, 128
tiles=(t1x1 t1x2 t2x1 t1x4 t2x2 t1x5 t2x4 t1x10 t2x5)
[ -d build ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON >/dev/null
cmake --build build -j >/dev/null
export OMP_PROC_BIND=close OMP_PLACES=cores
out=results/s1_tile_sweep.csv
mkdir -p results
echo "tile,accumulators,registers,layer,C,flops_per_cycle,pct_peak,pass" > "$out"
for t in "${tiles[@]}"; do
  kb=${t:1:1}; ow=${t#*x}; acc=$((kb * ow)); regs=$((acc + kb + 1))
  for L in "${layers[@]}"; do
    line=$(./build/bench_layer "data/tensors/b1/$L" "s1_$t" -r "$reps" --csv)
    IFS=, read -r _ _ _ C _ _ _ _ _ _ _ _ _ _ _ fpc pct _ _ pass <<< "$line"
    echo "$t,$acc,$regs,$L,$C,$fpc,$pct,$pass" >> "$out"
  done
done
python3 - "$out" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cs = sorted({int(r["C"]) for r in rows})
print(f"{'tile':<6} {'acc':>4} {'regs':>5}" + "".join(f"{'C=' + str(c):>9}" for c in cs) + "   FLOPs/cycle, 1 core")
for t in dict.fromkeys(r["tile"] for r in rows):
    rr = {int(r["C"]): r for r in rows if r["tile"] == t}
    first = next(iter(rr.values()))
    print(f"{t:<6} {first['accumulators']:>4} {first['registers']:>5}" + "".join(f"{float(rr[c]['flops_per_cycle']):>9.1f}" for c in cs))
PY
echo "-> $out"
