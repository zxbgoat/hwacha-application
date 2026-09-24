# Build / run the DeepBench cases, one directory per operation.
# DeepBench (baidu-research/DeepBench) benchmarks vendor libraries (cuDNN / cuBLAS / MKL ...) on the
# problem sets in code/kernels/*.h; it has no kernels of its own. Each <op>/ here holds <op>.cl (an
# OpenCL implementation of the operation with the library's semantics, written for hwacha-cc),
# <op>.s (hwacha-cc's assembly of it), <op>_main.c (bare-metal host: runs several problem shapes
# from the DeepBench sets, scaled down, against a scalar reference and prints <op> PASS/FAIL with
# cycle counts) and common.h.
#
#   make                 build every case            -> <op>/<op>.riscv
#   make <op>            build one case
#   make run             build + run all on Spike, one PASS/FAIL line each
#   make <op>.spike      run one case
#   make clean           remove the .riscv files
#   make gen-<op>        regenerate <op>/<op>.s from the .cl (clang -> hwacha-cc)
ROOT    ?= /home/tesla/hwacha-compiler
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
CLANG   := $(HOME)/miniforge3/bin/clang
HWCC    := $(ROOT)/hwacha-cc/build/hwacha-cc
SPIKE   := $(ROOT)/install/bin/spike --isa=rv64gc --extension=hwacha
CLFLAGS := -x cl -cl-std=CL1.2 -Xclang -finclude-default-header -Xclang -cl-ext=+cl_khr_fp64,+cl_khr_fp16 --target=riscv64-unknown-elf \
           -march=rv64gc -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d \
           -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
# per-case defines / hwacha-cc flags
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

# every sub-directory that contains <name>/<name>.cl is a case (gen-<name> compiles it); the ones that
# also have <name>.s -- i.e. that hwacha-cc takes -- are built and run
ALLCASES := $(sort $(foreach s,$(wildcard */*.cl),$(if $(filter $(notdir $(s)),$(patsubst %/,%,$(dir $(s))).cl),$(patsubst %/,%,$(dir $(s))))))
CASES := $(foreach c,$(ALLCASES),$(if $(wildcard $(c)/$(c).s),$(c)))
RISCV := $(foreach c,$(CASES),$(c)/$(c).riscv)

.PHONY: all run clean list $(ALLCASES) $(addsuffix .spike,$(ALLCASES))
all: $(RISCV)
list:
	@printf '%s\n' $(CASES)

define CASE_RULE
$(1): $(1)/$(1).riscv
$(1)/$(1).riscv: $(1)/$(1)_main.c $(1)/$(1).s $(1)/common.h $(wildcard $(1)/$(1)_ref.c) $(COMMON)/crt.S $(COMMON)/syscalls.c
	cd $(1) && $(CC) $(CFLAGS) -o $(1).riscv $(1)_main.c $(notdir $(wildcard $(1)/$(1)_ref.c)) $(1).s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
$(1).spike: $(1)/$(1).riscv
	$(SPIKE) $(1)/$(1).riscv
gen-$(1): $(1)/$(1).cl
	$(CLANG) $(CLFLAGS) $(CLDEFS_$(1)) $(1)/$(1).cl -o $(1)/$(1).ll && $(HWCC) $(HWCCFLAGS_$(1)) $(1)/$(1).ll -o $(1)/$(1).s && rm -f $(1)/$(1).ll
endef
$(foreach c,$(ALLCASES),$(eval $(call CASE_RULE,$(c))))

# build and run every case, one PASS/FAIL line each (the hosts print "<app> PASS" / "<app> FAIL")
run:
	@pass=0; fail=0; for c in $(CASES); do \
	  if $(MAKE) -s $$c/$$c.riscv >/tmp/db_$$c.log 2>&1; then \
	    r=$$($(SPIKE) $$c/$$c.riscv 2>&1 | grep -oE "^$$c[a-z_]* (PASS|FAIL)" | tail -1); \
	    echo "$${r:-$$c NO-OUTPUT}"; case "$$r" in *PASS) pass=$$((pass+1));; *) fail=$$((fail+1));; esac; \
	  else echo "$$c BUILD-FAIL (see /tmp/db_$$c.log)"; fail=$$((fail+1)); fi; \
	done; echo "---- $$pass passed, $$fail failed of $$(echo $(CASES) | wc -w)"
clean:
	rm -f $(RISCV)
