# NCHW8c vs NCHW vs no-halo on O'Malley's 2x5 stride-2 kernel (all 7 stride-2 layers)
#   make && taskset -c 0 ./layout_s2.x 10
layout_s2.x: layout_s2.c kernel.h
	gcc -O3 -march=broadwell -std=c99 -Wall -o $@ layout_s2.c -lm
clean:
	rm -f layout_s2.x
