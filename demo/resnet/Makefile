SHELL := /bin/bash
# ResNet-50 image classification on Hwacha: pretrained ImageNet weights, a real image, top-5 printed on Spike.
#   make gen [IMG=path.jpg]   export from PyTorch (resnet50.s, weights, check.bin, labels.h) -- needs the torch-mlir venv
#   make                      build resnet50.riscv
#   make run                  run on Spike
ROOT    ?= $(HOME)/hwacha-compiler
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
SPIKE   := $(ROOT)/install/bin/spike --isa=rv64gc --extension=hwacha
MEM     ?= 2048
IMG     ?= dog.jpg
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d \
           -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(CURDIR)/demo.ld
# no built-in rules: otherwise `%: %.S` would "build" resnet50_weights.bin from resnet50_weights.bin.S,
# overwriting the weight blob with a linked ELF
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.PHONY: all run gen clean
all: resnet50.riscv
crt.o: $(COMMON)/crt.S
	$(CC) $(CFLAGS) -c -o $@ $<
syscalls.o: $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<
main.o: main.c labels.h check.bin
	$(CC) $(CFLAGS) -DCHECK_FILE='"check.bin"' -DMEM_MB=$(MEM) -c -o $@ main.c
resnet50.riscv: main.o resnet50.s hwlib.s resnet50_weights.bin.S resnet50_weights.bin crt.o syscalls.o demo.ld
	$(CC) $(CFLAGS) -o $@ crt.o syscalls.o main.o resnet50.s hwlib.s resnet50_weights.bin.S $(LDFLAGS)
run: resnet50.riscv
	$(SPIKE) -m$(MEM) $<
gen:
	./gen.sh $(IMG)
clean:
	rm -f *.o *.riscv
