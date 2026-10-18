# 3×3 stride-1 convolution: design notes (Om)

This walks through `src/conv3x3_s1.cpp` in the order the Project 2 handout asks its questions, so it doubles as the outline for the mid-project document and slides. Every number here is checkable: shapes come from `docs/LAYERS.md`, instruction numbers from your Assignment 1 measurements or [uops.info](https://uops.info) (Broadwell column).

## 1. What the layer computes

For every image `n`, output channel `k` and output pixel `(oh, ow)`:

```
out[n][k][oh][ow] = SiLU( bias[k] + Σ_c Σ_r=0..2 Σ_s=0..2  in[n][c][oh+r-1][ow+s-1] · W[k][c][r][s] )
```

Stride 1 and pad 1 mean the output is the same size as the input. There are 32 such layers in yolov8n: 20 inside the C2f bottlenecks (C = K = 16, 32, 64, 128 at 160², 80², 40², 20²) and 12 in the Detect head (64 or 80 output channels, some with 128 or 256 inputs). Together they are 5.8 of the model's 8.7 GFLOP (66%), so this kernel is most of the network. BatchNorm is not part of the work: the ONNX export already folded it into `W` and `bias` (the exported graph has zero BatchNormalization nodes), which is exactly Prof. Low's point.

## 2. Kernel description (handout part 1)

**Independent operations.** Each output element is an independent dot product of length C·9. A batch-1 layer at 80×80 with 32 channels has 204,800 of them, so there is far more parallelism than the machine needs.

**Dependent instructions inside one.** The reduction: C·9 multiply-adds into one accumulator, each depending on the previous result. Written as FMAs, that is a chain of C·9 `vfmadd231ps` with a 5-cycle latency between links.

**Functional units.** The FMA units on ports 0 and 1 (two per core). Loads of weights and broadcasts of inputs use the load ports 2 and 3. The single store per output vector uses port 4. The SiLU epilogue uses the FMA/multiply units plus one divide.

**SIMD.** AVX2 with FMA3: 16 YMM registers, 8 FP32 lanes each. No AVX-512.

**Specialized units.** None that help FP32 on Broadwell: no AVX-512, no VNNI (and VNNI is int8 anyway), no AMX. F16C only converts half precision.

## 3. Peak (handout part 2)

| Instruction | Use | Latency | Throughput |
|---|---|---|---|
| `vfmadd231ps ymm` | the multiply-add | 5 cycles | 2 per cycle (ports 0, 1) |
| `vbroadcastss ymm, m32` | one input value to 8 lanes | load-to-use ~5 to 7 cycles | 2 per cycle (ports 2, 3, load-only uop) |
| `vmovaps ymm, m256` | 8 weights | load-to-use ~5 to 7 cycles | 2 per cycle (ports 2, 3) |
| `vmovaps m256, ymm` | output store | | 1 per cycle (port 4) |

Machine peak per core = 2 FMA/cycle × 8 lanes × 2 FLOPs = **32 FLOPs/cycle**. The node has 20 cores, so 640 FLOPs/cycle. Use your Assignment 1 latency and throughput numbers in the slides; they are your own measurements of exactly these instructions.

The kernel can reach that peak only if (a) at least 2 × 5 = **10 independent accumulators** are in flight, so the FMA latency is hidden, and (b) the loads it needs per FMA fit in the 2 load ports. Section 4 picks a tile that satisfies both. The epilogue is the only non-FMA work: about 20 vector ops per output register (exp polynomial, store) plus one `vdivps`, which on Broadwell occupies the divider for roughly 10 cycles. Against C·9 FMAs per output register (72 FMA cycles at C = 16, 576 at C = 128) that is on the order of 20% for the 16-channel layers and a few percent for the 128-channel ones. Replacing the divide with `vrcpps` plus one Newton step is an easy win for Simon. That is the kernel's theoretical peak per layer: 32 × FMA_cycles / (FMA_cycles + epilogue_cycles).

## 4. The microkernel (handout part 3)

**Tile choice.** The accumulator tile is `TILE_KB` output-channel blocks (8 channels each) by `TILE_OW` output pixels. Per input channel `ci` and kernel tap `(r, s)` the inner step loads `TILE_KB` weight vectors and broadcasts `TILE_OW` input values, then issues `TILE_KB × TILE_OW` FMAs.

| Tile | Accumulators | Registers (acc + weights + bcast) | Load uops per step | FMA cycles per step | Fits yolov8n shapes? |
|---|---|---|---|---|---|
| 1 × 10 | 10 | 12 | 11 → 5.5 cycles | 5 | load-bound, rejected |
| **2 × 5** | **10** | **13** | **7 → 3.5 cycles** | **5** | **yes: 160, 80, 40, 20 all divide by 5; every K/8 is even** |
| 2 × 6 | 12 | 15 | 8 → 4 cycles | 6 | needs a tail for every width (none divide by 6) |
| 3 × 4 | 12 | 16 | 7 | 6 | K/8 = 2, 4, 8, 16 not divisible by 3 |
| 4 × 3 | 12 | 17 | 7 | 6 | does not fit in 16 registers |

2 × 5 is the smallest tile that hides the FMA latency, stays FMA-bound rather than load-bound, and needs no edge code for any yolov8n layer. Its weak spot is that it sits exactly at the 10-accumulator minimum, so any extra stall shows up directly. If measurements show that, 2 × 6 with a tail is the next thing to try.

**Size of the kernel.** One call produces 2 × 8 channels × 5 pixels = **80 outputs**, reducing over all C·9 inputs before storing anything.

**Register allocation.** 10 accumulators (`c0[0..4]` for output block kb, `c1[0..4]` for kb+1), 2 weight registers (`wv0`, `wv1`), 1 broadcast register (`x`): 13 of the 16 YMM registers. `cmake --build build --target asm_s1` confirms the compiler kept all of them in registers (no `(%rsp)` traffic in the loop).

**Register reuse.** Each weight register is reused by 5 FMAs (one per output pixel). Each broadcast register is reused by 2 FMAs (one per output block). The accumulators live in registers for the whole C·9 reduction and are written to memory once. The bias is loaded once into the accumulators' initial value, so the bias add is free.

**Implementation.** Read `conv3x3_s1_v0` first: it is the same loop nest in scalar C, with `acc[8]` where v1 has a register. Then `ukr_2x5`:

```
acc[0..1][0..4] = bias for blocks kb, kb+1
for cb in 0..C/8-1:                     # input channel block
  for r in 0..2, s in 0..2:             # kernel tap
    for ci in 0..7:                     # channel inside the block (unrolled)
      wv0 = load W[kb  ][cb][r][s][ci][0..7]
      wv1 = load W[kb+1][cb][r][s][ci][0..7]
      for j in 0..4:                    # output pixel (unrolled)
        x = broadcast in[cb][oh+r][ow+j+s][ci]
        acc[0][j] += x * wv0;  acc[1][j] += x * wv1
store SiLU(acc) to out[kb..kb+1][oh][ow..ow+4]
```

The driver `conv3x3_s1_v1` calls this for every `(n, kb pair, oh, ow step of 5)`, with OpenMP over the first three loops.

## 5. Memory layout (handout part 4)

**Why NCHW8c.** Putting 8 channels innermost makes the 8 output-channel weights for one `(ci, r, s)` a single aligned 32-byte load, and the 5 input values one microkernel step needs (`ow .. ow+4`, same channel) sit 32 bytes apart in the same couple of cache lines. In plain NCHW the weights for 8 output channels are `C·9` floats apart and the kernel would gather.

**Why a halo.** The input buffer stores the zero padding explicitly, so the microkernel never checks image edges and every call is identical. The cost is a little memory per plane.

**One layout for the whole network.** If every kernel reads and writes NCHW8c with a 1-pixel halo (the team contract in `include/common.h`), layers chain with no conversions. That answers Prof. Low's second comment: the surrounding layers make several costs disappear. In NCHW8c each 8-channel block is a contiguous plane, so the C2f `Split` is a pointer offset and `Concat` is the producer writing into its slice of a larger buffer, both free. The Bottleneck residual `x + cv2(cv1(x))` can be fused into the second conv's epilogue (load x, add, store), which removes a full pass over the tensor. Those are good v2+ items and good slide material.

**Working set per microkernel call.**

| Input channels C | Input (C/8 × 3 rows × 7 px × 32 B) | Weights for 2 blocks (C/8 × 4.6 KB) | Lives in |
|---|---|---|---|
| 16 | 1.3 KB | 9.2 KB | L1 (32 KB) |
| 32 | 2.7 KB | 18.4 KB | L1 |
| 64 | 5.4 KB | 36.9 KB | weights spill to L2 (256 KB) |
| 128 | 10.8 KB | 73.7 KB | L2 |
| 256 | 21.5 KB | 147 KB | L2, barely |

Consecutive calls along a row reuse the same weights, so for C ≤ 32 everything stays in L1. From C = 64 up the weight slab streams from L2 on every call. Whether that hurts is the first thing to measure on ece007 (compare % of peak for the 16, 32, 64 and 128 channel layers). If it does, v2 blocks the `cb` loop: process a few input-channel blocks at a time so the weights stay in L1, carrying partial sums in the output buffer between blocks.

## 6. Parallelization and NUMA

OpenMP splits `(n, kb pair, oh)` statically. At batch 1 that is between 80 (the 20×20, 64-output layers) and 400 units, so 20 threads always have work, but on the 20×20 layers each unit is tiny and the fork/join overhead shows. Batch > 1 gives more work per layer. Two known issues to fix as you scale past one socket: `bench_layer` allocates and zero-fills buffers on one thread, so all memory lands on socket 0 (first-touch); and threads 11 to 20 then read across QPI. Initializing buffers inside an `omp parallel for` with the same schedule as the kernel fixes the first.

## 7. Measuring honestly

Report FLOPs/cycle in core cycles for one thread and in TSC cycles (wall time) for many threads; `bench_layer` prints which one it used. Over 100% of peak means the cycle count is wrong (turbo running the core faster than the 2.4 GHz TSC), not that the kernel is magic. The baseline numbers from `scripts/baseline_bench.py` use wall time at 2.4 GHz, so compare against our TSC-cycle numbers.

## 8. Next steps, in order

1. On ece007: `scripts/run_kernels.sh s1_v1 3x3s1` and `python scripts/baseline_bench.py`, then `plot_results.py`. These are the plots for handout part 5.
2. Check single-core % of peak across channel counts (section 5 table) to see where L2-resident weights cost you.
3. v2: cache-block the `cb` loop for C ≥ 64.
4. Scale threads 1, 2, 5, 10, 20 at batch 1 and 16; fix first-touch for 20.
5. Fuse the Bottleneck residual add into `model.*.m.*.cv2`'s epilogue (needs a dump of the Add output and a harness flag).
