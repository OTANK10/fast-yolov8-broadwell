import subprocess
import json
import argparse

BINARY = './kernel_driver.x'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, default=20)
    parser.add_argument('-b', '--binary', type=str, required=False, default=BINARY,
                        help='driver to run, e.g. ./kernel_v3.x')
    parser.add_argument('-l', '--layer', type=str, required=False, default=None,
                        help='run only this layer, e.g. model.3, or "sweep" for the k sweep')
    parser.add_argument('-o', '--out', type=str, required=False, default='./data/flops.json')
    args = parser.parse_args()

    cmd = [args.binary] + ([args.layer] if args.layer else [])

    # layer name -> list of FLOPs/cycle, one per run
    run_times = {}
    failed = set()

    for _ in range(args.n):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
        lines = result.stdout.decode('utf-8').splitlines()

        for line in lines:
            x = line.split()
            name = x[0]
            flops = float(x[6])
            correct = int(x[9])

            if not correct:
                failed.add(name)
            run_times.setdefault(name, []).append(flops)

    for name, flops in run_times.items():
        print(name)
        print(flops)

    if failed:
        print(f"WARNING: incorrect output on {sorted(failed)}")

    with open(args.out, 'w') as f:
        json.dump(run_times, f, indent=4)


if __name__ == '__main__':
    main()
