#!/usr/bin/env bash
# "Know your machine": prints the numbers the report needs, in our terms.
#   bash scripts/machine_info.sh
set -uo pipefail
model=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
sockets=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')
cores=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')
tpc=$(lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
numa=$(lscpu | awk -F: '/^NUMA node\(s\)/{gsub(/ /,"",$2); print $2}')
flags=$(grep -m1 "^flags" /proc/cpuinfo)
has() { grep -qw "$1" <<< "$flags" && echo yes || echo no; }
ghz=$(sed -n 's/.*@ *\([0-9.]*\)GHz.*/\1/p' <<< "$model")
echo "CPU             $model"
echo "sockets x cores ${sockets} x ${cores}  (threads per core: ${tpc}, NUMA nodes: ${numa})"
echo "base clock      ${ghz:-unknown} GHz (TSC rate; turbo can be higher)"
echo "AVX2 / FMA      $(has avx2) / $(has fma)"
echo "AVX-512F        $(has avx512f)"
lscpu | grep -E "^L1d|^L2|^L3" | sed 's/  */ /g'
echo
if [ "$(has avx2)" = yes ] && [ "$(has fma)" = yes ]; then
  echo "AVX2 FP32 peak per core = FMA units x 8 lanes x 2 = 32 FLOPs/cycle if the core has 2 FMA units"
  echo "  (every Intel core from Haswell on has 2 for 256-bit FMA; check this model on ark.intel.com / uops.info)."
fi
if [ "$(has avx512f)" = yes ]; then
  echo "This CPU also has AVX-512: its own peak is 16 lanes x 2 x (#512-bit FMA units, 1 or 2 depending on the model)."
  echo "Our kernels are AVX2, so report % of the AVX2 peak (export PEAK_FPC=32) and mention the AVX-512 headroom."
fi
