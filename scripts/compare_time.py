#!/usr/bin/env python3
"""Wall-clock comparison of kernels vs baselines, layer by layer (microseconds).

    python scripts/compare_time.py results/s1_v2_b1_t1.csv results/baseline_b1_t1.csv

FLOPs/cycle is not comparable between our kernels (core cycles, which follow turbo)
and the baselines (wall time divided by the 2.4 GHz base clock). Time is.
"""
import csv, sys
kern = list(csv.DictReader(open(sys.argv[1])))
base = {r["layer"]: r for r in csv.DictReader(open(sys.argv[2]))}
name = kern[0]["kernel"]
print(f"{'layer':<24} {'C':>4} {'HxW':>8} {name+' us':>10} {'OpenCV us':>10} {'ORT us':>9} {'vs OpenCV':>10} {'vs ORT':>8}")
tot = [0.0, 0.0, 0.0]
for r in kern:
    b = base.get(r["layer"])
    if not b or "time_us" not in r:
        continue
    t, o, x = float(r["time_us"]), float(b["ocv_us"]), float(b["ort_us"])
    tot = [tot[0] + t, tot[1] + o, tot[2] + x]
    print(f"{r['layer']:<24} {r['C']:>4} {r['H']+'x'+r['W']:>8} {t:>10.0f} {o:>10.0f} {x:>9.0f} {o/t:>9.2f}x {x/t:>7.2f}x")
print(f"{'total':<24} {'':>4} {'':>8} {tot[0]:>10.0f} {tot[1]:>10.0f} {tot[2]:>9.0f} {tot[1]/tot[0]:>9.2f}x {tot[2]/tot[0]:>7.2f}x")
print("(speedup > 1 means our kernel is faster)")
