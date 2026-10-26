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

# layer sizes for the x labels: (C in, K out, input H = W)
LAYERS = {
    'model.0':  (3,   16,  640),
    'model.1':  (16,  32,  320),
    'model.3':  (32,  64,  160),
    'model.5':  (64,  128, 80),
    'model.7':  (128, 256, 40),
    'model.16': (64,  64,  80),
    'model.19': (128, 128, 40),
}


def layer_size(name):
    C, K, H = LAYERS[name]
    return f"{C}→{K} ch\n{H}×{H}"


def label_for(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace('layers_', '').replace('sweep_', '')
    return BLOCKS.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='*', default=['./data/flops.json'],
                        help='one or more flops.json files to compare')
    args = parser.parse_args()

    plt.figure(figsize=(10, 5.5))
    n_sets = len(args.data)
    width = 0.8 / n_sets

    for i, path in enumerate(args.data):
        with open(path, 'r') as f:
            data = json.load(f)

        names = list(data.keys())
        medians = [statistics.median(v) for v in data.values()]
        xs = [j + (i - (n_sets - 1) / 2) * width for j in range(len(names))]
        plt.bar(xs, medians, width=width, label=label_for(path))

        print(label_for(path))
        for name, flops in data.items():
            med = statistics.median(flops)
            print(f"  {name:10s} median {med:6.2f}  max {max(flops):6.2f}  "
                  f"min {min(flops):6.2f}  ({100 * med / PEAK:4.1f}% of peak)")

    plt.axhline(PEAK, linestyle='--', color='gray', label='peak (32)')
    ticks = [layer_size(n) if n in LAYERS else n for n in names]
    plt.xticks(range(len(names)), ticks)
    plt.ylabel("FLOPs per cycle (median)")
    plt.xlabel("Layer: input → output channels, input size")
    plt.title("3x3 Stride 2 Convolution: FLOPs per Cycle per Layer")
    plt.ylim(0, PEAK + 6)
    plt.legend(loc='upper center', ncol=n_sets + 1, frameon=False)
    plt.tight_layout()
    plot_path = f'./plots/flops_per_layer_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
