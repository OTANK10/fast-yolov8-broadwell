
CC		= gcc
KERNEL		?= kernel_v2
CC_FLAGS 	= -mavx2 -mfma -O3 -std=c99
LIBS		= -lm
RM 		= rm -f

default: all assemble

all: kernel


kernel: kernel_driver.x
	./kernel_driver.x

kernel_driver.x: kernel_driver.c kernel.c kernel.h
	$(CC) $(CC_FLAGS) -DKERNEL_FN=$(KERNEL) -o kernel_driver.x kernel_driver.c kernel.c $(LIBS)


# one binary per kernel for the k sweep: kernel_v2.x, kernel_v3.x, kernel_v4.x
SWEEP_KERNELS	= kernel_v2 kernel_v3 kernel_v4
RUNS		?= 20

kernel_v%.x: kernel_driver.c kernel.c kernel.h
	$(CC) $(CC_FLAGS) -DKERNEL_FN=kernel_v$* -o $@ kernel_driver.c kernel.c $(LIBS)

sweep: $(SWEEP_KERNELS:=.x)
	mkdir -p data plots
	for k in $(SWEEP_KERNELS); do python3 collect_data.py -n $(RUNS) -b ./$$k.x -l sweep -o data/sweep_$$k.json; done
	python3 plot_sweep.py $(SWEEP_KERNELS:%=data/sweep_%.json)

# all 7 YOLOv8n 3x3 stride-2 layers, one bar per kernel per layer
layers: $(SWEEP_KERNELS:=.x)
	mkdir -p data plots
	for k in $(SWEEP_KERNELS); do python3 collect_data.py -n $(RUNS) -b ./$$k.x -o data/layers_$$k.json; done
	python3 plot_data.py $(SWEEP_KERNELS:%=data/layers_%.json)

assemble: kernel_driver.x
	objdump -s -d -f --source ./kernel_driver.x > kernel_driver.S

clean:
	rm -rf *.x *.S
