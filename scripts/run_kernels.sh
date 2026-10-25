#!/usr/bin/env bash
# Run one kernel over every layer of one kind and collect a CSV.
#   scripts/run_kernels.sh s1_v1 3x3s1            # batch 1, 1 thread
#   scripts/run_kernels.sh s1_v1 3x3s1 16 20      # batch 16, 20 threads
#   EXTRA=--no-silu scripts/run_kernels.sh s1_v10 3x3s1   # extra bench_layer flags; tag goes in the file name
# kinds: 3x3s1 (Om)  3x3s2 (O'Malley)  1x1 (Simon)   -> results/<kernel>_b<N>_t<T>.csv
set -euo pipefail
kernel=$1; kind=$2; batch=${3:-1}; threads=${4:-1}; reps=${REPS:-20}
data=data/tensors/b$batch
extra=${EXTRA:-}
tag=$(echo "$extra" | tr -d ' -' )
out=results/${kernel}_b${batch}_t${threads}${tag:+_$tag}.csv
mkdir -p results
[ -d build ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_ENABLE_BROADWELL=ON >/dev/null
cmake --build build -j >/dev/null
# bind threads to cores, fill socket 0 first (NUMA: keep data and threads on one socket while T<=10)
export OMP_PROC_BIND=close OMP_PLACES=cores
echo "layer,kernel,N,C,H,W,K,R,S,stride,threads,flops,best_cycles,median_cycles,cycle_src,flops_per_cycle,pct_peak,gflops,max_err,pass,time_us" > "$out"
fail=0
while IFS=, read -r dir onnx_name k _; do
  [[ "$k" == "$kind" ]] || continue
  if ! ./build/bench_layer "$data/$dir" "$kernel" -r "$reps" -t "$threads" $extra --csv >> "$out"; then
    echo "FAIL: $dir" >&2; fail=1
  fi
done < <(tail -n +2 "$data/layers.csv")
awk -F, '{printf "%-26s %-7s %-6s %-4s %-4s %-12s %-8s %-6s\n", $1, $3"x"$4, $5"x"$6, $7, $10, $16, $17, $20}' "$out"
[[ $fail == 0 ]] && echo "all layers PASS -> $out" || { echo "some layers FAILED"; exit 1; }
