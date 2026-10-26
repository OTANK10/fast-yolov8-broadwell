3x3 stride-2 conv kernel (YOLOv8n downsampling layers), hw2-style harness.

You can run the code using make.

  kernel.c          your kernel (starts as a correct scalar placeholder)
  kernel.h          conv_desc_t + layout helpers, same layouts as the team repo
  kernel_driver.c   naive NCHW reference, layout conversion, rdtsc timing, check

./kernel_driver.x [layer] prints one line per layer:
  name  C  K  H  W  cycles  FLOPs/cycle  %peak  max_abs_err  correct

collect_data.py runs the binary -n times and saves FLOPs/cycle to data/flops.json
  python3 collect_data.py -n 20 [-l model.3] [-o data/v2.json]
plot_data.py plots median FLOPs/cycle per layer (pass several json files to compare)
  python3 plot_data.py data/v1.json data/v2.json

Setup for plotting: python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
Measure on ece028 only (Broadwell, peak 32 fp32 FLOPs/cycle/core).
