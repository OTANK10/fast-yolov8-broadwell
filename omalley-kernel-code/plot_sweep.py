import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import statistics
from datetime import datetime
import argparse

PEAK = 32.0  # fp32 FLOPs/cycle/core: 2 FMA ports x 8 lanes x 2
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# register block of each kernel, for the legend
BLOCKS = {'kernel_v2': '2x5', 'kernel_v3': '2x6', 'kernel_v4': '3x4'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+',
                        help='sweep json files from  collect_data.py -l sweep, one per kernel')
    args = parser.parse_args()

    plt.figure(figsize=(9, 5))

    for path in args.data:
        with open(path, 'r') as f:
            data = json.load(f)

        # keys are k = C*3*3
        ks = sorted(int(k) for k in data)
        medians = [statistics.median(data[str(k)]) for k in ks]

        name = os.path.splitext(os.path.basename(path))[0].replace('sweep_', '')
        label = BLOCKS.get(name, name)
        plt.plot(ks, medians, marker='o', markersize=3, label=label)

        print(label)
        for k, med in zip(ks, medians):
            print(f"  k={k:5d}  median {med:6.2f}  ({100 * med / PEAK:4.1f}% of peak)")

    plt.axhline(PEAK, linestyle='--', color='gray', label='peak (32)')
    plt.ylabel("FLOPs per cycle")
    plt.xlabel("k = (Input Channels) * 3*3")
    plt.title("3x3 (Stride 2) Convolution: FLOPs per cycle vs k")
    plt.ylim(0, PEAK + 2)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = f'./plots/flops_vs_k_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
